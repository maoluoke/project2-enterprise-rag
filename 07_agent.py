"""Step 7：企业知识库智能体（Function Calling + MCP）。

决策模型：阿里百炼 deepseek-v4-flash-0731（免费档，OpenAI 兼容端点，原生 tool_calls）
能力：
  1. kb_search（Function Calling）——项目2的混合检索知识库，内部问题优先走这里
  2. tavily-search（MCP）——联网搜索，知识库没有/时效性问题自动补充外网
  3. 闲聊与常识问题直接回答，不浪费检索

注意：qwen-agent 对 deepseek 模型会回退到提示词式 function calling（不稳定），
因此这里用 OpenAI 兼容端点的原生 tool_calls + mcp 客户端手写 ReAct 循环。

用法：
    python 07_agent.py "公司 multipart 上传的默认限制是多少？"
    python 07_agent.py                          # 交互模式

环境变量：
    DASHSCOPE_API_KEY   百炼 key（~/.bashrc 已配置）
    TAVILY_API_KEY      Tavily 联网搜索 key（~/.bashrc 已配置）
"""

import asyncio
import json
import os
import sys

import kb_tools  # noqa: F401  注册工具 + 保证 embedding 最先导入
import retrieval
import skill_loader
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AGENT_MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash-0731")
MAX_TOOL_ROUNDS = 10


def _load_env_from_bashrc(name: str) -> str:
    """从 ~/.bashrc 兜底读取环境变量（IDE/调试器启动时不会加载 bashrc）。"""
    value = os.getenv(name, "")
    if value:
        return value
    bashrc = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"export {name}="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant for Redwood Inference.\n"
    "Decision policy:\n"
    "1. For ANY question about internal projects, incidents, policies, people, "
    "tickets, code, emails or meetings, ALWAYS call the kb_search tool first. "
    "Never answer internal questions from memory.\n"
    "2. If kb_search returns found=false or the passages clearly cannot answer "
    "the question, use tavily web search to supplement, and state the answer "
    "came from the public web rather than internal documents.\n"
    "3. For chit-chat or general knowledge, answer directly without tools.\n"
    "4. When using kb_search results, cite doc_id in the form [doc_id].\n"
    "5. When the user's request matches a skill listed below, call load_skill "
    "with the skill name FIRST to get detailed instructions, then follow them.\n"
    "6. Answer in the same language as the user's question. Be concise."
)

LOAD_SKILL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Load the full instruction set of a skill by name. Call this FIRST "
            "when the user's request matches a skill's description, then follow "
            "the returned instructions to complete the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill name, e.g. kb-research"},
            },
            "required": ["name"],
        },
    },
}

KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "kb_search",
        "description": (
            "Search the internal enterprise knowledge base of Redwood Inference "
            "(Slack, Gmail, Google Drive, Linear, GitHub, Jira, Confluence). "
            "Use FIRST for internal questions. Returns passages with doc_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "standalone search query"},
                "top_k": {"type": "integer", "description": "default 5"},
            },
            "required": ["query"],
        },
    },
}


def call_kb_search(arguments: dict) -> str:
    query = arguments.get("query", "")
    top_k = int(arguments.get("top_k", 5))
    results, _ = retrieval.search(query, mode="hybrid_rerank", top_k=top_k)
    if not results:
        return json.dumps({"found": False, "message": "No relevant internal documents."},
                          ensure_ascii=False)
    items = [{
        "doc_id": r["doc_id"],
        "source_type": r["source_type"],
        "title": r["title"][:120],
        "passage": (r.get("matched_passage") or r.get("content") or "")[:800],
    } for r in results[:top_k]]
    return json.dumps({"found": True, "passages": items}, ensure_ascii=False)


class Agent:
    """OpenAI 原生 tool_calls + Tavily MCP 的 ReAct 智能体。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=_load_env_from_bashrc("DASHSCOPE_API_KEY"),
            base_url=DASHSCOPE_BASE_URL,
            timeout=120,
        )
        self.tool_trace = []

    async def _get_tavily_tools(self, session: ClientSession):
        """把 MCP server 的工具描述转换成 OpenAI tools schema。"""
        listed = await session.list_tools()
        schemas = []
        for t in listed.tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"mcp__{t.name}",
                    "description": t.description or t.name,
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return schemas

    async def run(self, query: str):
        self.tool_trace = []
        base_tools = [KB_TOOL_SCHEMA, LOAD_SKILL_TOOL_SCHEMA]
        tavily_key = _load_env_from_bashrc("TAVILY_API_KEY")
        if not tavily_key:
            # 无 Tavily key：降级为纯知识库模式，不启动 MCP
            return await self._run_loop(query, base_tools, None)

        server = StdioServerParameters(
            command="npx",
            args=["-y", "tavily-mcp@0.1.4"],
            env={**os.environ, "TAVILY_API_KEY": tavily_key},
        )
        try:
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tavily_tools = await self._get_tavily_tools(session)
                    tools = base_tools + tavily_tools
                    return await self._run_loop(query, tools, session)
        except BaseException as e:
            # MCP 连接失败（如网络/启动错误）：降级为纯知识库模式重试
            print(f"[警告] Tavily MCP 不可用（{type(e).__name__}），降级为仅知识库模式")
            self.tool_trace = []
            return await self._run_loop(query, base_tools, None)

    async def _run_loop(self, query: str, tools: list, session):
        system_prompt = SYSTEM_PROMPT + "\n\n" + skill_loader.skills_prompt_section()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or ""

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.tool_trace.append(name)

                if name == "kb_search":
                    result = call_kb_search(args)
                elif name == "load_skill":
                    result = skill_loader.load_skill(args.get("name", ""))
                elif name.startswith("mcp__") and session is not None:
                    out = await session.call_tool(name[5:], args)
                    result = "\n".join(
                        getattr(p, "text", str(p)) for p in out.content
                    )[:4000]
                else:
                    result = f"Unknown tool: {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
        return "（达到最大工具调用轮数，未能得出最终答案）"


def run_query(query: str):
    agent = Agent()
    answer = asyncio.run(agent.run(query))
    trace = agent.tool_trace
    if trace:
        print(f"[工具调用链] {' -> '.join(trace)}")
    else:
        print("[工具调用链] 未调用任何工具（直接回答）")
    print(f"\n回答:\n{answer}")


def main():
    queries = [a for a in sys.argv[1:] if not a.startswith("--")]
    if queries:
        for q in queries:
            run_query(q)
    else:
        print("进入智能体模式（输入 q 退出）")
        while True:
            q = input("\n请输入问题: ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                run_query(q)


if __name__ == "__main__":
    main()
