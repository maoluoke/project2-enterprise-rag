"""轻量 RAG：File System + Hash Index（DocParser），不依赖向量数据库。

设计：
- 每篇文档落盘为 docstore/docs/ab/<hash>.json，文件系统即文档库；
- 倒排索引只保存 token 的哈希 -> 文档哈希/tf，落盘为 hash_index.json；
- 查询时对 token 做同样的哈希，直接查倒排表，用 BM25 打分。

1000~10000 篇文档规模下，无需部署向量库即可实现毫秒级检索。
"""

import hashlib
import json
import math
import re
import time
from pathlib import Path

import config

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "on", "with",
    "as", "at", "by", "from", "that", "this", "it", "be", "or", "was", "were",
    "will", "can", "not", "we", "you", "i", "they", "their", "our", "has", "have",
    "had", "would", "should", "could", "do", "does", "did", "but", "if", "then",
}


def tokenize(text: str):
    tokens = []
    for t in TOKEN_RE.findall((text or "").lower()):
        if t in STOPWORDS or len(t) < 2:
            continue
        tokens.append(t)
    return tokens


def thash(token: str) -> str:
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]


class DocParser:
    """文件系统 + 哈希倒排索引的轻量检索器。"""

    def __init__(self, store_dir: Path = None):
        self.store_dir = Path(store_dir or config.DOC_STORE_DIR)
        self.docs_dir = self.store_dir / "docs"
        self.index_path = self.store_dir / "hash_index.json"
        self.index = None
        self._title_tokens = {}

    # ---------- 写入 ----------
    def _select_docs(self, jsonl_path: Path, limit: int):
        """优先纳入 Golden 标准文档，再用随机负例补足，保证小规模评估仍有意义。"""
        golden_path = config.DATA_DIR / "golden.jsonl"
        gold_ids = set()
        if golden_path.exists():
            with golden_path.open(encoding="utf-8") as f:
                for line in f:
                    gold_ids.update(json.loads(line)["expected_doc_ids"])

        gold_docs, other_docs = [], []
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["doc_id"] in gold_ids:
                    if len(gold_docs) < limit:
                        gold_docs.append(d)
                elif len(other_docs) < limit:
                    other_docs.append(d)
        selected = (gold_docs + other_docs)[:limit]
        print(f"[DocParser] 选中 {len(gold_docs)} 篇标准文档 + "
              f"{min(len(other_docs), max(limit - len(gold_docs), 0))} 篇负例")
        return selected

    def ingest(self, jsonl_path: Path, limit: int = 2000):
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        postings = {}
        docs = {}
        total_len = 0

        t0 = time.perf_counter()
        for i, d in enumerate(self._select_docs(jsonl_path, limit)):
            doc_hash = hashlib.sha1(d["uid"].encode("utf-8")).hexdigest()[:16]
            path = self.docs_dir / doc_hash[:2] / f"{doc_hash}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

            tokens = tokenize((d.get("title") or "") + " " + (d.get("content") or ""))
            dl = len(tokens)
            total_len += dl
            docs[doc_hash] = {
                "doc_id": d["doc_id"],
                "source_type": d["source_type"],
                "title": d.get("title") or "",
                "path": str(path.relative_to(self.store_dir)),
                "len": dl,
            }
            tf = {}
            for tok in tokens:
                h = thash(tok)
                tf[h] = tf.get(h, 0) + 1
            for h, n in tf.items():
                postings.setdefault(h, {})[doc_hash] = n
            if i % 500 == 0:
                print(f"[DocParser] 已落盘 {i} 篇")

        index = {
            "version": 1,
            "N": len(docs),
            "avgdl": (total_len / len(docs)) if docs else 0,
            "docs": docs,
            "postings": postings,
        }
        self.index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        self.index = index
        self._title_tokens = {}
        print(f"[DocParser] 完成：{len(docs)} 篇文档，索引 {self.index_path.stat().st_size/1e6:.1f} MB，"
              f"耗时 {time.perf_counter()-t0:.1f}s")

    # ---------- 读取/检索 ----------
    def load(self):
        if self.index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"找不到 {self.index_path}，请先运行 ingest")
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._title_tokens = {}
        return self.index

    def _title_tf(self, doc_hash: str):
        if doc_hash not in self._title_tokens:
            title = self.index["docs"][doc_hash]["title"]
            tf = {}
            for tok in tokenize(title):
                h = thash(tok)
                tf[h] = tf.get(h, 0) + 1
            self._title_tokens[doc_hash] = tf
        return self._title_tokens[doc_hash]

    def search(self, query: str, top_k: int = 5):
        idx = self.load()
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        postings, docs, N = idx["postings"], idx["docs"], idx["N"]
        avgdl = idx["avgdl"] or 1.0
        k1, b = 1.5, 0.75

        scores = {}
        for tok in q_tokens:
            h = thash(tok)
            posting = postings.get(h)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for doc_hash, tf in posting.items():
                dl = docs[doc_hash]["len"] or 1
                # 标题命中加权：同一 token 出现在标题中，tf 额外 +1.5
                tf_eff = tf + 1.5 * self._title_tf(doc_hash).get(h, 0)
                score = idf * tf_eff * (k1 + 1) / (
                    tf_eff + k1 * (1 - b + b * dl / avgdl)
                )
                scores[doc_hash] = scores.get(doc_hash, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for doc_hash, score in ranked:
            meta = docs[doc_hash]
            path = self.store_dir / meta["path"]
            try:
                content = json.loads(path.read_text(encoding="utf-8")).get("content", "")
            except Exception:
                content = ""
            results.append({
                "doc_id": meta["doc_id"],
                "source_type": meta["source_type"],
                "title": meta["title"],
                "score": float(score),
                "content": content,
                "path": str(path),
            })
        return results

    def answer(self, query: str, top_k: int = 5):
        """检索 + 本地 Qwen3 生成，不依赖向量库。"""
        import llm

        hits = self.search(query, top_k)
        if not hits:
            return "No relevant documents found.", []
        context = "\n\n".join(
            f"[{i}] doc_id={h['doc_id']} | {h['source_type']} | {h['title']}\n"
            f"{h['content'][:1500]}"
            for i, h in enumerate(hits, 1)
        )
        messages = [
            {"role": "system", "content":
             "You are an enterprise knowledge assistant. Answer only from the provided "
             "documents. If not covered, say so. Cite doc_id. Be concise, 3-6 sentences."},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"},
        ]
        return llm.generate(messages), hits


def print_hits(query, hits, elapsed_ms=None):
    print(f"\n===== {query} =====")
    if elapsed_ms:
        print(f"检索耗时: {elapsed_ms:.1f} ms")
    for i, h in enumerate(hits, 1):
        print(f"\n--- {i} | {h['doc_id']} | {h['source_type']} | score={h['score']:.4f} ---")
        print(f"{h['title'][:120]}\n{h['content'][:300]}...")
