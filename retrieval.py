"""混合检索核心：BM25 + 向量加权融合（或 RRF），可选 BGE-Reranker 精排。

支持 4 种模式，便于评估阶段做消融对比：
- bm25          纯关键词
- vector        纯语义
- hybrid        BM25 + 向量融合
- hybrid_rerank 融合召回后再用交叉编码器精排
"""

import time

import config
import es_client
from embedding import embed_query
from reranker import rerank_documents

MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")


def _candidate(hit: dict) -> dict:
    src = hit.get("_source", {})
    return {
        "uid": hit.get("_id"),
        "doc_id": src.get("doc_id"),
        "source_type": src.get("source_type"),
        "title": src.get("title") or "",
        "content": src.get("content") or "",
        "bm25_score": 0.0,
        "vector_score": 0.0,
        "score": 0.0,
    }


def _dedupe_doc_id(candidates):
    """同一 doc_id 可能存在多个版本（数据集中有 4 组重复 id），保留最高分版本。"""
    best = {}
    for c in candidates:
        key = c["doc_id"]
        if key not in best or c["score"] >= best[key]["score"]:
            best[key] = c
    return list(best.values())


def fuse_weighted(bm25_hits, vector_hits, alpha: float, recall_k: int):
    """加权融合：两路分数各自按本查询最高分归一化到 [0,1]。"""
    bm25_scores = [h["_score"] for h in bm25_hits] or [0.0]
    vec_scores = [h["_score"] for h in vector_hits] or [0.0]
    bm25_max = max(bm25_scores) or 1.0
    vec_max = max(vec_scores) or 1.0

    fused = {}
    for hit in bm25_hits:
        c = _candidate(hit)
        c["bm25_score"] = float(hit["_score"] / bm25_max)
        c["score"] = (1 - alpha) * c["bm25_score"]
        fused[c["doc_id"]] = c

    for hit in vector_hits:
        c = _candidate(hit)
        c["vector_score"] = float(max(hit["_score"], 0.0) / vec_max)
        if c["doc_id"] in fused:
            fused[c["doc_id"]]["vector_score"] = c["vector_score"]
            fused[c["doc_id"]]["score"] += alpha * c["vector_score"]
        else:
            c["score"] = alpha * c["vector_score"]
            fused[c["doc_id"]] = c

    ranked = sorted(fused.values(), key=lambda x: -x["score"])[:recall_k]
    return _dedupe_doc_id(ranked)


def fuse_rrf(bm25_hits, vector_hits, recall_k: int, rrf_k: int = 60):
    """Reciprocal Rank Fusion：只依赖名次，避免两路分数量纲不一致。"""
    fused = {}
    for hits, field in ((bm25_hits, "bm25_score"), (vector_hits, "vector_score")):
        for rank, hit in enumerate(hits, 1):
            cand = _candidate(hit)
            c = fused.get(cand["doc_id"])
            if c is None:
                c = cand
                fused[c["doc_id"]] = c
            c[field] = max(c[field], float(hit["_score"]))
            c["score"] += 1.0 / (rrf_k + rank)
    ranked = sorted(fused.values(), key=lambda x: -x["score"])[:recall_k]
    return _dedupe_doc_id(ranked)


def search(query: str, mode: str = "hybrid_rerank", top_k: int = None,
           recall_k: int = None, alpha: float = None, source_types=None,
           min_score: float = None, fusion: str = None):
    """执行一次检索，返回 (results, timings)。

    results 的每项包含 doc_id/title/content/score/matched_passage 等字段。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode}")
    top_k = top_k or config.FINAL_K
    recall_k = recall_k or config.RECALL_K
    alpha = config.HYBRID_ALPHA if alpha is None else alpha
    fusion = fusion or config.FUSION_METHOD
    min_score = config.MIN_RERANK_SCORE if min_score is None else min_score

    es = es_client.get_es()
    timings = {}

    bm25_hits, vector_hits = [], []
    if mode in ("bm25", "hybrid", "hybrid_rerank"):
        t = time.perf_counter()
        bm25_hits = es_client.bm25_search(es, query, size=recall_k, source_types=source_types)
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000

    if mode in ("vector", "hybrid", "hybrid_rerank"):
        t = time.perf_counter()
        qv = embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        vector_hits = es_client.vector_search(
            es, qv, k=recall_k, num_candidates=recall_k * 2, source_types=source_types
        )
        timings["knn_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if mode == "bm25":
        candidates = [_candidate(h) for h in bm25_hits]
        for c, h in zip(candidates, bm25_hits):
            c["bm25_score"] = float(h["_score"])
            c["score"] = float(h["_score"])
        candidates = _dedupe_doc_id(candidates)[:top_k]
    elif mode == "vector":
        candidates = [_candidate(h) for h in vector_hits]
        for c, h in zip(candidates, vector_hits):
            c["vector_score"] = float(h["_score"])
            c["score"] = float(h["_score"])
        candidates = _dedupe_doc_id(candidates)[:top_k]
    else:
        if fusion == "rrf":
            candidates = fuse_rrf(bm25_hits, vector_hits, recall_k, config.RRF_K)
        else:
            candidates = fuse_weighted(bm25_hits, vector_hits, alpha, recall_k)
    timings["fuse_ms"] = (time.perf_counter() - t) * 1000

    if mode == "hybrid_rerank" and candidates:
        t = time.perf_counter()
        results = rerank_documents(query, candidates, final_k=top_k)
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000
    else:
        results = candidates[:top_k]
        for item in results:
            item.setdefault("matched_passage", (item.get("content") or "")[:800])

    if min_score > 0:
        results = [r for r in results if r.get("score", 0) >= min_score]

    return results, timings


def format_results(query: str, results, timings=None):
    print(f"\n===== 查询: {query} =====")
    if timings:
        print("耗时: " + " | ".join(f"{k}={v:.1f}ms" for k, v in timings.items()))
    if not results:
        print("未检索到相关文档")
        return
    for rank, r in enumerate(results, 1):
        print(f"\n--- 排名 {rank} | score={r.get('score', 0):.4f} | {r['source_type']} | {r['doc_id']} ---")
        print(f"标题: {r['title'][:120]}")
        print(f"片段: {(r.get('matched_passage') or r.get('content'))[:260]}...")
