"""数据准备：EnterpriseRAG-Bench parquet -> 项目内 JSONL。

- 文档：保留 doc_id 作为业务主键；同一 doc_id 可能有多个版本，用 uid
  （doc_id + 内容哈希前 10 位）作为索引唯一键，避免 ES 覆盖。
- 采样：默认取 2 万篇，但强制包含全部 Golden 问题对应的 722 篇标准答案文档，
  保证检索评估仍然覆盖全部 500 道题。
- Golden Dataset：默认全部 500 题，也可按题型分层抽样 50+。
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

import config


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def prepare(sample_size: int = None, golden_size: int = None, seed: int = None):
    config.ensure_dirs()
    sample_size = config.SAMPLE_SIZE if sample_size is None else sample_size
    golden_size = config.GOLDEN_SIZE if golden_size is None else golden_size
    seed = config.RANDOM_SEED if seed is None else seed

    doc_path = config.RAW_DOCS_PATH
    q_path = config.RAW_QUESTIONS_PATH
    if not doc_path.exists() or not q_path.exists():
        raise FileNotFoundError(f"找不到原始数据: {doc_path} / {q_path}")

    print(f"[Data] 读取原始文档 {doc_path}")
    docs = pd.read_parquet(doc_path)
    docs["doc_id"] = docs["doc_id"].astype(str)
    docs["source_type"] = docs["source_type"].astype(str)
    docs["title"] = docs["title"].fillna("").astype(str)
    docs["content"] = docs["content"].fillna("").astype(str)
    docs["content_len"] = docs["content"].str.len()
    docs["uid"] = [
        f"{did}#{_sha1(content)[:10]}" for did, content in zip(docs.doc_id, docs.content)
    ]
    docs = docs.drop_duplicates(subset=["uid"]).reset_index(drop=True)

    questions = pd.read_parquet(q_path)
    gold_ids = {x for lst in questions.expected_doc_ids for x in _as_list(lst)}
    print(f"[Data] 原始文档 {len(docs)} 篇（去重 uid 后） | Golden 标准文档 {len(gold_ids)} 篇")

    if sample_size and sample_size > 0 and sample_size < len(docs):
        gold_docs = docs[docs.doc_id.isin(gold_ids)]
        rest = docs[~docs.doc_id.isin(gold_ids)]
        need = max(sample_size - len(gold_docs), 0)
        sampled = rest.sample(n=min(need, len(rest)), random_state=seed)
        docs = pd.concat([gold_docs, sampled], ignore_index=True)
        print(f"[Data] 样例模式：{len(gold_docs)} 篇标准文档 + {len(sampled)} 篇随机负例")
    else:
        print("[Data] 全量模式：写入全部文档")

    docs = docs.sort_values("uid").reset_index(drop=True)
    docs_path = config.DATA_DIR / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for row in docs.itertuples(index=False):
            f.write(json.dumps({
                "uid": row.uid,
                "doc_id": row.doc_id,
                "source_type": row.source_type,
                "title": row.title,
                "content": row.content,
                "content_len": int(row.content_len),
            }, ensure_ascii=False) + "\n")
    print(f"[Data] 已写入 {len(docs)} 篇文档 -> {docs_path}")

    if golden_size and golden_size > 0 and golden_size < len(questions):
        per_type = max(1, golden_size // questions.question_type.nunique())
        parts = [
            g.sample(n=min(per_type, len(g)), random_state=seed)
            for _, g in questions.groupby("question_type", sort=False)
        ]
        golden = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
        golden = golden.head(golden_size).reset_index(drop=True)
        print(f"[Data] Golden Dataset 分层抽样 {len(golden)} 题")
    else:
        golden = questions.reset_index(drop=True)
        print(f"[Data] Golden Dataset 使用全部 {len(golden)} 题")

    golden_path = config.DATA_DIR / "golden.jsonl"
    with golden_path.open("w", encoding="utf-8") as f:
        for row in golden.itertuples(index=False):
            f.write(json.dumps({
                "question_id": row.question_id,
                "question_type": row.question_type,
                "source_types": _as_list(row.source_types),
                "question": row.question,
                "expected_doc_ids": _as_list(row.expected_doc_ids),
                "gold_answer": row.gold_answer,
                "answer_facts": _as_list(row.answer_facts),
            }, ensure_ascii=False) + "\n")
    print(f"[Data] 已写入 {len(golden)} 道 Golden 问题 -> {golden_path}")

    manifest = {
        "source": "EnterpriseRAG-Bench (onyx-dot-app)",
        "sample_size": int(len(docs)),
        "full_corpus_size": int(len(pd.read_parquet(doc_path, columns=["doc_id"]))),
        "golden_questions": int(len(golden)),
        "gold_docs": int(len(gold_ids)),
        "gold_docs_in_sample": int(docs.doc_id.isin(gold_ids).sum()),
        "source_counts": docs.source_type.value_counts().to_dict(),
    }
    manifest_path = config.DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Data] manifest -> {manifest_path}")
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="准备 EnterpriseRAG-Bench 文档与 Golden 问题")
    parser.add_argument("--sample", type=int, default=None, help="样例文档数；0 表示全量")
    parser.add_argument("--golden", type=int, default=None, help="Golden 问题数；0 表示全部")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    prepare(args.sample, args.golden, args.seed)


if __name__ == "__main__":
    main()
