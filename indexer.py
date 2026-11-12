"""ES 索引构建：第一阶段全量 BM25，第二阶段分批写入向量。

两阶段设计的好处：
1. BM25 索引很快，先让系统可用；
2. 向量生成是瓶颈（Qwen3-Embedding-4B 约 38 docs/s），支持断点续跑，
   中断后从 checkpoint 继续，不重复消耗 GPU 时间。
"""

import json
import os
from pathlib import Path

from tqdm import tqdm

import config
import es_client
from embedding import build_doc_text, embed_documents

CHECKPOINT_PATH = config.DATA_DIR / "vector_checkpoint.json"


def stream_docs(path: Path, limit: int = None):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)


def build_bm25(es, docs_path, limit=None):
    """全量 BM25 索引。"""
    es.indices.put_settings(
        index=config.ES_DOC_INDEX,
        body={"index": {"refresh_interval": "-1"}},
    )
    count = 0

    def gen():
        nonlocal count
        for d in stream_docs(docs_path, limit):
            count += 1
            yield {
                "_op_type": "index",
                "_index": config.ES_DOC_INDEX,
                "_id": d["uid"],
                "_source": {
                    "uid": d["uid"],
                    "doc_id": d["doc_id"],
                    "source_type": d["source_type"],
                    "title": d["title"],
                    "content": d["content"],
                    "content_len": d["content_len"],
                },
            }

    print("[Index] 阶段1：写入 BM25 文档...")
    for success, info in _streaming(es, gen()):
        if not success and info:
            print(f"[Index] 写入失败: {str(info)[:160]}")
        if count % 2000 == 0:
            print(f"[Index] 已提交 {count} 篇")

    es.indices.put_settings(
        index=config.ES_DOC_INDEX,
        body={"index": {"refresh_interval": "30s"}},
    )
    es.indices.refresh(index=config.ES_DOC_INDEX)
    print(f"[Index] BM25 完成，索引文档数 {es_client.count_docs(es)}")


def _streaming(es, actions):
    from elasticsearch import helpers

    yield from helpers.streaming_bulk(
        es, actions, chunk_size=500, max_chunk_bytes=50 * 1024 * 1024,
        raise_on_error=False, request_timeout=180,
    )


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            return int(json.loads(CHECKPOINT_PATH.read_text())["processed"])
        except Exception:
            return 0
    return 0


def save_checkpoint(processed: int):
    CHECKPOINT_PATH.write_text(json.dumps({"processed": processed}), encoding="utf-8")


def build_vectors(es, docs_path, limit=None, resume=True):
    """阶段2：逐批生成向量并 update 到 ES，支持断点续跑。"""
    start = load_checkpoint() if resume else 0
    if start:
        print(f"[Index] 从 checkpoint 继续：已处理 {start} 篇")

    es.indices.put_settings(
        index=config.ES_DOC_INDEX,
        body={"index": {"refresh_interval": "-1"}},
    )

    processed = 0
    batch_docs, batch_texts = [], []

    def flush():
        nonlocal processed, batch_docs, batch_texts
        if not batch_docs:
            return
        vectors = embed_documents(batch_texts)
        actions = []
        for d, vec in zip(batch_docs, vectors):
            actions.append({
                "_op_type": "update",
                "_index": config.ES_DOC_INDEX,
                "_id": d["uid"],
                "doc": {"embedding": list(map(float, vec))},
                "doc_as_upsert": True,
            })
        for success, info in _streaming(es, actions):
            if not success and info:
                print(f"[Index] 向量写入失败: {str(info)[:160]}")
        processed += len(batch_docs)
        save_checkpoint(start + processed)
        batch_docs, batch_texts = [], []

    total = limit or sum(1 for _ in docs_path.open(encoding="utf-8"))
    pbar = tqdm(total=max(total - start, 0), desc="向量化", unit="doc")
    for i, d in enumerate(stream_docs(docs_path, limit)):
        if i < start:
            continue
        text = build_doc_text(d["title"], d["content"])
        if not text:
            processed += 1
            save_checkpoint(start + processed)
            pbar.update(1)
            continue
        batch_docs.append(d)
        batch_texts.append(text)
        if len(batch_docs) >= config.EMBED_BATCH:
            flush()
            pbar.update(config.EMBED_BATCH)
    flush()
    pbar.close()

    es.indices.put_settings(
        index=config.ES_DOC_INDEX,
        body={"index": {"refresh_interval": "30s"}},
    )
    es.indices.refresh(index=config.ES_DOC_INDEX)
    docs_with_vec = es_client.count_docs(es, with_vectors=True)
    print(f"[Index] 向量阶段完成：已处理 {start + processed} 篇，含向量文档 {docs_with_vec}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="构建 ES 混合索引")
    parser.add_argument("--recreate", action="store_true", help="删除并重建索引")
    parser.add_argument("--skip-vectors", action="store_true", help="只建 BM25")
    parser.add_argument("--vectors-only", action="store_true", help="只补向量")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--restart-vectors", action="store_true", help="忽略 checkpoint 重新向量化")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少篇（0=全部）")
    parser.add_argument("--shards", type=int, default=int(os.getenv("ES_SHARDS", "1")))
    args = parser.parse_args()

    docs_path = config.DATA_DIR / "docs.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"缺少 {docs_path}，请先运行 01_prepare_data.py")

    es = es_client.get_es()
    limit = args.limit or None
    if not args.vectors_only:
        es_client.create_doc_index(es, recreate=args.recreate, shards=args.shards)
        build_bm25(es, docs_path, limit)
    if not args.skip_vectors:
        build_vectors(es, docs_path, limit, resume=not args.restart_vectors)


if __name__ == "__main__":
    main()
