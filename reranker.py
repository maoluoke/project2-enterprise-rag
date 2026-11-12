"""BGE-Reranker 精排：长文档先切窗，再对 query-passage 逐对打分取最大值。

单篇文档平均约 4800 字符，而交叉编码器最大输入 512 token。
如果直接把整篇文档截断送入，答案往往落在被截掉的尾部。
因此这里对全文均匀切窗，逐窗打分后取该文档的最高分，并保留命中窗口用于生成。
"""

import os
import re

import config

_reranker = None


def load_reranker():
    """懒加载 bge-reranker-base；torch 在此处才导入，确保 embedding 已先行加载。"""
    global _reranker
    if _reranker is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = config.RERANK_MODEL_DIR
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model = model.half()
        model.to(device).eval()
        _reranker = (tokenizer, model, device)
        print(f"[Reranker] 已加载 bge-reranker-base（device={device}）")
    return _reranker


def split_passages(text: str, window: int = None, stride: int = None, max_n: int = None):
    """把长文本切成均匀窗口；窗口数超过 max_n 时在全文中均匀抽样（含首尾）。"""
    window = window or config.RERANK_WINDOW
    stride = stride or config.RERANK_STRIDE
    max_n = max_n or config.RERANK_MAX_PASSAGES
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return [""]
    if len(text) <= window:
        return [text]

    passages = []
    start = 0
    while start < len(text):
        passages.append(text[start:start + window])
        if start + window >= len(text):
            break
        start += stride

    if len(passages) > max_n:
        # 均匀抽样，保证文档尾部也有机会被精排命中
        idxs = np_linspace(0, len(passages) - 1, max_n)
        passages = [passages[i] for i in idxs]
    return passages


def np_linspace(a, b, n):
    """不依赖 numpy 的整数均匀取点。"""
    if n <= 1:
        return [a]
    span = b - a
    return [a + round(span * i / (n - 1)) for i in range(n)]


def score_pairs(query: str, passages) -> list:
    """批量计算 query 与 passage 的相关性概率。"""
    if not passages:
        return []
    import torch

    tokenizer, model, device = load_reranker()
    probs = []
    batch = 32
    with torch.inference_mode():
        for i in range(0, len(passages), batch):
            pairs = [[query, p] for p in passages[i:i + batch]]
            inputs = tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits.view(-1).float()
            probs.extend(torch.sigmoid(logits).tolist())
    return probs


def rerank_documents(query: str, candidates, final_k: int = None):
    """对候选文档逐窗打分，取每篇最高窗口分；返回按分降序的 Top-K。"""
    if not candidates:
        return []
    final_k = final_k or config.FINAL_K

    pairs, owners = [], []
    for idx, cand in enumerate(candidates):
        title = (cand.get("title") or "").strip()
        text = (cand.get("content") or "").strip()
        for passage in split_passages(text):
            pairs.append((title + " " + passage).strip() if title else passage)
            owners.append((idx, passage))

    probs = score_pairs(query, pairs)

    best = {}
    for (idx, passage), prob in zip(owners, probs):
        if idx not in best or prob > best[idx][0]:
            best[idx] = (prob, passage)

    ranked = sorted(
        ((best[i][0], i, best[i][1]) for i in range(len(candidates)) if i in best),
        key=lambda x: -x[0],
    )
    results = []
    for prob, idx, passage in ranked[:final_k]:
        item = dict(candidates[idx])
        item["score"] = float(prob)
        item["matched_passage"] = passage
        results.append(item)
    return results
