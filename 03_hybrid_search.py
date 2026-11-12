"""Step 3：混合检索命令行（BM25 + Vector + BGE-Reranker）。

用法：
    python 03_hybrid_search.py "Who caused the API gateway 5xx surge?"
    python 03_hybrid_search.py --mode bm25 "..."
    python 03_hybrid_search.py --mode hybrid_rerank --alpha 0.5 "..."
    python 03_hybrid_search.py                # 交互模式
"""

import sys

import retrieval


def main():
    args = sys.argv[1:]
    mode, alpha, top_k, source_types, fusion = "hybrid_rerank", None, 5, None, None
    queries = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--mode":
            mode = args[i + 1]
            i += 2
        elif a == "--alpha":
            alpha = float(args[i + 1])
            i += 2
        elif a == "--top-k":
            top_k = int(args[i + 1])
            i += 2
        elif a == "--source":
            source_types = args[i + 1].split(",")
            i += 2
        elif a == "--rrf":
            fusion = "rrf"
            i += 1
        else:
            queries.append(a)
            i += 1

    def run(query):
        results, timings = retrieval.search(
            query, mode=mode, top_k=top_k, alpha=alpha,
            source_types=source_types, fusion=fusion,
        )
        retrieval.format_results(query, results, timings)

    if queries:
        for q in queries:
            run(q)
    else:
        print(f"进入检索模式（mode={mode}，输入 q 退出）")
        while True:
            q = input("\n请输入查询: ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                run(q)


if __name__ == "__main__":
    main()
