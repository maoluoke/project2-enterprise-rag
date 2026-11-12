"""Qwen3-Embedding-4B 嵌入封装。

注意：本模块必须在任何 import torch 之前导入。
sentence_transformers 会先加载 datasets/pyarrow，若 torch 先导入会导致
libstdc++ 的 CXXABI 版本冲突（本机 pyarrow 与 torch 共存时的已知问题）。
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model = None


def load_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedding] 加载 {config.EMBED_MODEL}（{config.EMBED_DIM} 维）...")
        _model = SentenceTransformer(
            config.EMBED_MODEL,
            device=os.getenv("EMBED_DEVICE", "cuda"),
            model_kwargs={"torch_dtype": "bfloat16"},
            truncate_dim=config.EMBED_DIM,
        )
        _model.max_seq_length = config.EMBED_MAX_LEN
    return _model


def embed_documents(texts, batch_size=None, show_progress_bar=False):
    """文档侧不加 query 指令，返回 float32 向量列表。"""
    if not texts:
        return []
    model = load_embedder()
    vecs = model.encode(
        list(texts),
        batch_size=batch_size or config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str):
    """查询侧使用 Qwen3-Embedding 自带的 query 指令 prompt。"""
    model = load_embedder()
    vec = model.encode(text, prompt_name="query", normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")


def build_doc_text(title: str, content: str) -> str:
    """用于生成向量的文本：标题 + 正文前若干字符，控制 token 成本。"""
    title = (title or "").strip()
    content = (content or "").strip()
    return (title + "\n" + content[: config.EMBED_MAX_CHARS]).strip()
