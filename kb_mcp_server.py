"""MCP Server：把项目2的知识库能力暴露给任何 MCP 客户端（FastMCP）。

用法：
    python kb_mcp_server.py          # stdio 模式，供 qwen-agent / Claude 等调用

提供的工具：
- kb_search       混合检索（BM25+向量+精排）
- kb_stats        知识库规模统计
"""

import json

import embedding  # noqa: F401  必须先于 torch 相关导入
import es_client
import retrieval
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Enterprise RAG Knowledge Base")


@mcp.tool()
def kb_search(query: str, top_k: int = 5) -> str:
    """Search the internal enterprise knowledge base (Slack, Gmail, Drive,
    Linear, GitHub, Jira, Confluence). Use for questions about internal
    projects, incidents, policies, people, tickets, code or meetings.
    Returns top passages with doc_id for citation."""
    results, _ = retrieval.search(query, mode="hybrid_rerank", top_k=top_k)
    if not results:
        return "No relevant documents found in the internal knowledge base."
    lines = []
    for i, r in enumerate(results, 1):
        passage = (r.get("matched_passage") or r.get("content") or "")[:800]
        lines.append(
            f"[{i}] doc_id={r['doc_id']} | source={r['source_type']} | "
            f"score={r.get('score', 0):.4f}\n{r['title'][:120]}\n{passage}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def kb_stats() -> str:
    """Get statistics of the knowledge base: total documents, documents with
    vector embeddings, and source type breakdown."""
    es = es_client.get_es()
    total = es_client.count_docs(es)
    with_vec = es_client.count_docs(es, with_vectors=True)
    agg = es.search(
        index=es_client.config.ES_DOC_INDEX,
        body={
            "size": 0,
            "aggs": {"by_source": {"terms": {"field": "source_type", "size": 20}}},
        },
    )
    sources = {
        b["key"]: b["doc_count"]
        for b in agg["aggregations"]["by_source"]["buckets"]
    }
    return json.dumps({
        "total_docs": total,
        "docs_with_vector": with_vec,
        "source_types": sources,
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
