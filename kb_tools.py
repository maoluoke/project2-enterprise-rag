"""Function Calling 工具集：把项目2的混合检索能力封装成模型可调用的工具。

模型（deepseek-v4-flash）通过 function calling 自主决定：
- 用户问的是内部知识库问题 -> 调用 kb_search
- 知识库里没有 -> 调用联网搜索（Tavily MCP）
- 闲聊/常识 -> 直接回答

依赖注意：必须先 import embedding，再让 qwen-agent 加载，避免 pyarrow/torch 冲突。
"""

import json

import embedding  # noqa: F401  必须最先导入
import retrieval
from qwen_agent.tools.base import BaseTool, register_tool


def _format_hits(results, top_k):
    """把检索结果压缩成适合 function 返回值的 JSON 文本。"""
    items = []
    for r in results[:top_k]:
        passage = (r.get("matched_passage") or r.get("content") or "")[:800]
        items.append({
            "doc_id": r["doc_id"],
            "source_type": r["source_type"],
            "title": r["title"][:120],
            "score": round(float(r.get("score", 0)), 4),
            "passage": passage,
        })
    return items


@register_tool("kb_search")
class KBSearchTool(BaseTool):
    """企业知识库检索工具：BM25 + 向量混合召回 + BGE 精排。"""

    description = (
        "Search the internal enterprise knowledge base of Redwood Inference "
        "(Slack, Gmail, Google Drive, Linear, GitHub, Jira, Confluence documents). "
        "Use this tool FIRST for any question about internal projects, incidents, "
        "policies, people, tickets, code, or meeting content. Returns the top "
        "relevant passages with doc_id citations."
    )
    parameters = [{
        "name": "query",
        "type": "string",
        "description": "A standalone English search query with key terms, error codes, dates preserved.",
        "required": True,
    }, {
        "name": "top_k",
        "type": "integer",
        "description": "Number of passages to return, default 5.",
        "required": False,
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        query = args["query"]
        top_k = int(args.get("top_k", 5))
        results, timings = retrieval.search(
            query, mode="hybrid_rerank", top_k=top_k
        )
        if not results:
            return json.dumps({
                "found": False,
                "message": "No relevant documents in the internal knowledge base.",
            }, ensure_ascii=False)
        return json.dumps({
            "found": True,
            "query": query,
            "passages": _format_hits(results, top_k),
        }, ensure_ascii=False)
