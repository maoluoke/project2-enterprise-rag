"""Step 6：自动化检索评估流水线。

Golden Dataset 来自 EnterpriseRAG-Bench 的 500 道真实业务问题（含标准文档 ID），
对 4 种检索策略做消融，输出 HitRate / MRR / Recall / 延迟，并落盘 JSON+CSV。

用法：
    python 06_evaluate.py --limit 50
    python 06_evaluate.py --modes bm25,vector,hybrid,hybrid_rerank
    python 06_evaluate.py --limit 0 --top-k 5
"""

import argparse
import json
import os
import time

import config
import retrieval
from eval_utils import hit_at_k, mrr_at_k, percentile, recall_at_k

MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]


def load_golden(limit: int = 0):
    rows = []
    with (config.DATA_DIR / "golden.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def evaluate_mode(rows, mode, top_k, recall_k, alpha, fusion):
    metrics = {"hit": 0, "mrr": 0.0, "recall": 0.0, "latencies": []}
    answerable = 0
    no_answer = 0
    no_answer_empty = 0
    per_type = {}

    for i, row in enumerate(rows, 1):
        expected = row["expected_doc_ids"]
        t = time.perf_counter()
        results, _ = retrieval.search(
            row["question"], mode=mode, top_k=top_k, recall_k=recall_k,
            alpha=alpha, fusion=fusion,
        )
        elapsed = (time.perf_counter() - t) * 1000
        metrics["latencies"].append(elapsed)

        retrieved = [r["doc_id"] for r in results]
        stats = per_type.setdefault(row["question_type"], {"n": 0, "hit": 0, "mrr": 0.0, "recall": 0.0})

        if expected:
            answerable += 1
            h = hit_at_k(retrieved, expected, top_k)
            m = mrr_at_k(retrieved, expected, top_k)
            r = recall_at_k(retrieved, expected, top_k)
            metrics["hit"] += h
            metrics["mrr"] += m
            metrics["recall"] += r
            stats["n"] += 1
            stats["hit"] += h
            stats["mrr"] += m
            stats["recall"] += r
        else:
            no_answer += 1
            if not retrieved:
                no_answer_empty += 1

        if i % 50 == 0:
            print(f"  [{mode}] {i}/{len(rows)}")

    n = answerable or 1
    lat = metrics["latencies"]
    return {
        "mode": mode,
        "questions": len(rows),
        "answerable": answerable,
        "hit_rate": metrics["hit"] / n,
        "mrr": metrics["mrr"] / n,
        "recall": metrics["recall"] / n,
        "no_answer_questions": no_answer,
        "no_answer_empty_rate": (no_answer_empty / no_answer) if no_answer else 0.0,
        "latency_p50_ms": percentile(lat, 0.50),
        "latency_p95_ms": percentile(lat, 0.95),
        "per_type": {
            t: {
                "n": v["n"],
                "hit_rate": v["hit"] / v["n"] if v["n"] else 0.0,
                "mrr": v["mrr"] / v["n"] if v["n"] else 0.0,
                "recall": v["recall"] / v["n"] if v["n"] else 0.0,
            }
            for t, v in per_type.items()
        },
    }


def maybe_log_langfuse(report, top_k):
    """可选：把指标写入 LangFuse，未配置/未安装时自动跳过。"""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("[LangFuse] 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，跳过上传")
        return
    try:
        from langfuse import Langfuse

        client = Langfuse()
        for mode, m in report["modes"].items():
            client.create_score(
                name=f"hit_rate@{top_k}",
                value=m["hit_rate"],
                comment=f"mode={mode} mrr={m['mrr']:.4f} recall={m['recall']:.4f}",
            )
        client.flush()
        print("[LangFuse] 指标已上传")
    except Exception as e:
        print(f"[LangFuse] 上传失败（不影响本地报告）: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG 检索效果评估")
    parser.add_argument("--limit", type=int, default=0, help="评估题数，0=全部")
    parser.add_argument("--modes", type=str, default=",".join(MODES))
    parser.add_argument("--top-k", type=int, default=config.FINAL_K)
    parser.add_argument("--recall-k", type=int, default=config.RECALL_K)
    parser.add_argument("--alpha", type=float, default=config.HYBRID_ALPHA)
    parser.add_argument("--fusion", type=str, default=config.FUSION_METHOD)
    parser.add_argument("--output", type=str, default=str(config.DATA_DIR / "eval_report.json"))
    parser.add_argument("--langfuse", action="store_true")
    args = parser.parse_args()

    rows = load_golden(args.limit)
    print(f"[Eval] Golden 问题 {len(rows)} 道 | top_k={args.top_k} recall_k={args.recall_k}")

    report = {"top_k": args.top_k, "recall_k": args.recall_k, "alpha": args.alpha,
              "fusion": args.fusion, "modes": {}}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"\n[Eval] 模式 {mode} ...")
        report["modes"][mode] = evaluate_mode(
            rows, mode, args.top_k, args.recall_k, args.alpha, args.fusion
        )

    print("\n" + "=" * 78)
    print(f"{'Mode':<16}{'HitRate':>9}{'MRR':>9}{'Recall':>9}{'p50(ms)':>10}{'p95(ms)':>10}")
    print("-" * 78)
    for mode, m in report["modes"].items():
        print(f"{mode:<16}{m['hit_rate']:>9.4f}{m['mrr']:>9.4f}{m['recall']:>9.4f}"
              f"{m['latency_p50_ms']:>10.1f}{m['latency_p95_ms']:>10.1f}")
    print("=" * 78)

    # 计算相对提升，直接对应项目描述中的“Top-5 召回率提升”
    base = report["modes"].get("bm25") or report["modes"].get("vector")
    best = report["modes"].get("hybrid_rerank")
    if base and best:
        rel = (best["hit_rate"] - base["hit_rate"]) / base["hit_rate"] * 100 if base["hit_rate"] else 0
        report["relative_gain_vs_base_pct"] = round(rel, 2)
        print(f"hybrid_rerank 相对 {base['mode']} HitRate 提升: {rel:+.1f}%")

    out = os.path.abspath(args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    csv_path = os.path.splitext(out)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mode,hit_rate,mrr,recall,latency_p50_ms,latency_p95_ms\n")
        for mode, m in report["modes"].items():
            f.write(f"{mode},{m['hit_rate']:.4f},{m['mrr']:.4f},{m['recall']:.4f},"
                    f"{m['latency_p50_ms']:.1f},{m['latency_p95_ms']:.1f}\n")
    print(f"[Eval] 报告 -> {out}\n[Eval] 汇总 -> {csv_path}")

    if args.langfuse:
        maybe_log_langfuse(report, args.top_k)


if __name__ == "__main__":
    main()
