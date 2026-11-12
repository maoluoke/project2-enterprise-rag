"""Step 4：企业知识库 RAG 问答（LangChain + 分级记忆 + 混合检索）。

链路：
分级记忆读取 -> 多轮 Query 改写 -> BM25+向量混合召回 -> BGE 精排
-> LangChain Prompt/LLM 链生成 -> 写回 Redis 短期记忆与 ES 长期记忆

注意 import 顺序：embedding 必须先于 llm/reranker，避免 pyarrow 的 CXXABI 冲突。

用法：
    python 04_rag_answer.py "What is the default max_file_size for multipart uploads?"
    python 04_rag_answer.py --session s1 --user alice "What about total request size?"
    python 04_rag_answer.py                  # 交互模式
"""

import re
import sys
import time

import embedding  # noqa: F401  （必须先导入）
import config
import llm
import memory
import retrieval
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

SYSTEM_PROMPT = """You are the enterprise knowledge assistant for Redwood Inference.
Answer the user's question using ONLY the retrieved internal documents.

Rules:
1. If the documents do not contain the answer, say clearly that the available documents do not cover it.
2. When documents conflict, prefer the most recent/updated document and explicitly note the conflict.
3. Be concise: 3-6 sentences, no bullet lists unless the user asks for a procedure.
4. Cite the supporting document IDs in the form [doc_id] at the end of the answer.
5. Answer in the same language as the user's question.
"""

REWRITE_PROMPT = """Rewrite the user's latest question into a standalone search query for an enterprise
document search engine. Resolve pronouns using the conversation. Keep product names,
codename, error codes, dates and numbers exactly as written. Return ONLY the rewritten
query, no explanation, no quotes."""

_chain = None


def _to_dict_messages(prompt_value):
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    return [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in prompt_value.to_messages()
    ]


def get_chain():
    """LangChain Runnable：Prompt -> Qwen3 -> 纯文本。"""
    global _chain
    if _chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{user_input}"),
        ])
        _chain = (
            prompt
            | RunnableLambda(lambda pv: llm.generate(_to_dict_messages(pv)))
            | StrOutputParser()
        )
    return _chain


def rewrite_query(query, history):
    """多轮对话下把省略/指代查询改写为可独立检索的 query。"""
    if not history:
        return query
    turns = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        text = llm.generate(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{turns}\n\nLatest question: {query}"},
            ],
            max_new_tokens=96,
        )
        text = re.sub(r"^[\s\"']+|[\s\"']+$", "", text).split("\n")[0].strip()
        return text or query
    except Exception:
        return query


def build_user_input(query, results, long_term_text="（无）"):
    parts = [f"[Relevant past memory]\n{long_term_text}", "", "[Retrieved documents]"]
    for i, r in enumerate(results, 1):
        passage = (r.get("matched_passage") or r.get("content") or "")[:1200]
        parts.append(
            f"[{i}] doc_id={r['doc_id']} | source={r['source_type']} | title={r['title'][:160]}\n{passage}"
        )
    parts.append("")
    parts.append(f"Question: {query}")
    return "\n".join(parts)


def answer_question(query: str, mem: memory.MemoryManager = None,
                    mode: str = "hybrid_rerank", top_k: int = None,
                    do_rewrite: bool = True):
    """完整 RAG 单轮问答，返回 dict。"""
    t0 = time.perf_counter()
    history, long_term_text, long_hits = [], "（无）", []
    if mem:
        history, long_term_text, long_hits = mem.build_context(query)

    search_query = rewrite_query(query, history) if (mem and do_rewrite) else query

    t = time.perf_counter()
    results, timings = retrieval.search(search_query, mode=mode, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t) * 1000

    if not results:
        answer = "The available internal documents do not contain enough information to answer this question."
    else:
        user_input = build_user_input(query, results, long_term_text)
        answer = get_chain().invoke({"history": history, "user_input": user_input})

    if mem:
        mem.add_turn(query, answer)

    return {
        "answer": answer,
        "sources": [
            {
                "doc_id": r["doc_id"],
                "source_type": r["source_type"],
                "title": r["title"],
                "score": r.get("score"),
            }
            for r in results
        ],
        "rewritten_query": search_query,
        "memory_hits": [
            {"role": h["role"], "content": h["content"][:120], "score": h["score"]}
            for h in long_hits
        ],
        "timings": {**timings, "retrieval_total_ms": retrieval_ms},
        "total_ms": (time.perf_counter() - t0) * 1000,
    }


def print_answer(query, resp):
    print(f"\n===== 问题: {query} =====")
    if resp["rewritten_query"] != query:
        print(f"[改写] {resp['rewritten_query']}")
    print(f"\n回答:\n{resp['answer']}")
    if resp["sources"]:
        print("\n引用:")
        for i, s in enumerate(resp["sources"], 1):
            print(f"  [{i}] {s['doc_id']} | {s['source_type']} | {s['title'][:100]} | score={s['score']:.4f}")
    print(f"\n耗时: 检索 {resp['timings'].get('retrieval_total_ms', 0):.0f}ms | 总 {resp['total_ms']:.0f}ms")


def main():
    args = sys.argv[1:]
    user_id, session_id = "default", "default"
    mode, top_k, use_memory, do_rewrite = "hybrid_rerank", None, True, True
    reset, queries = False, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--user":
            user_id = args[i + 1]; i += 2
        elif a == "--session":
            session_id = args[i + 1]; i += 2
        elif a == "--mode":
            mode = args[i + 1]; i += 2
        elif a == "--top-k":
            top_k = int(args[i + 1]); i += 2
        elif a == "--no-memory":
            use_memory = False; i += 1
        elif a == "--no-rewrite":
            do_rewrite = False; i += 1
        elif a == "--reset":
            reset = True; i += 1
        else:
            queries.append(a); i += 1

    mem = memory.MemoryManager(user_id=user_id, session_id=session_id) if use_memory else None
    if mem and reset:
        mem.clear()
        print(f"[Memory] 已清空 session={session_id} 的短期记忆")

    if queries:
        for q in queries:
            print_answer(q, answer_question(q, mem, mode, top_k, do_rewrite))
    else:
        print(f"进入问答模式（user={user_id}, session={session_id}，输入 q 退出）")
        while True:
            q = input("\n请输入问题: ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                print_answer(q, answer_question(q, mem, mode, top_k, do_rewrite))


if __name__ == "__main__":
    main()
