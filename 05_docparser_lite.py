"""Step 5：File System + Hash Index 轻量 RAG（DocParser），无需向量数据库。

用法：
    python 05_docparser_lite.py --ingest 2000              # 摄入 2000 篇文档
    python 05_docparser_lite.py --ask "What caused the 5xx surge?"
    python 05_docparser_lite.py --benchmark 100            # 100 题检索延迟/HitRate
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import config
from docparser import DocParser, print_hits


def load_golden(limit: int = 100):
    path = config.DATA_DIR / "golden.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def benchmark(dp: DocParser, n: int = 100, top_k: int = 5):
    rows = load_golden(n)
    latencies, hits, rr = [], 0, []
    for row in rows:
        t = time.perf_counter()
        results = dp.search(row["question"], top_k)
        latencies.append((time.perf_counter() - t) * 1000)
        expected = set(row["expected_doc_ids"])
        if not expected:
            continue
        found = [r["doc_id"] for r in results]
        if expected.intersection(found):
            hits += 1
        for rank, doc_id in enumerate(found, 1):
            if doc_id in expected:
                rr.append(1.0 / rank)
                break

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    answerable = sum(1 for r in load_golden(n) if r["expected_doc_ids"])
    print(f"\n[Benchmark] {len(rows)} 题 | 可评估 {answerable} 题")
    print(f"HitRate@{top_k}: {hits/answerable:.4f} | MRR@{top_k}: {sum(rr)/answerable:.4f}")
    print(f"检索延迟 p50={p50:.2f}ms p95={p95:.2f}ms | 索引文档 {dp.load()['N']} 篇")


def main():
    parser = argparse.ArgumentParser(description="轻量 DocParser RAG")
    parser.add_argument("--ingest", type=int, default=0, help="摄入文档数，默认 2000")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--ask", nargs="*", default=None)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dp = DocParser()
    need_ingest = args.rebuild or not dp.index_path.exists()
    if args.ingest or need_ingest:
        limit = args.ingest or 2000
        print(f"[DocParser] 摄入 {limit} 篇文档到 {dp.store_dir}")
        dp.ingest(config.DATA_DIR / "docs.jsonl", limit=limit)
    else:
        dp.load()
        print(f"[DocParser] 已加载索引：{dp.index['N']} 篇文档")

    if args.benchmark:
        benchmark(dp, args.benchmark, args.top_k)

    if args.ask:
        for q in args.ask:
            t = time.perf_counter()
            hits = dp.search(q, args.top_k)
            print_hits(q, hits, (time.perf_counter() - t) * 1000)
            answer, _ = dp.answer(q, args.top_k)
            print(f"\n回答:\n{answer}")


if __name__ == "__main__":
    main()
