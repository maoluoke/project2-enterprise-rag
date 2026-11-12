"""检索评估指标：HitRate / MRR / Recall。"""


def dedupe(ids):
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def hit_at_k(retrieved, expected, k):
    return 1.0 if set(dedupe(retrieved)[:k]) & set(expected) else 0.0


def mrr_at_k(retrieved, expected, k):
    expected = set(expected)
    for rank, doc_id in enumerate(dedupe(retrieved)[:k], 1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved, expected, k):
    expected = set(expected)
    if not expected:
        return 0.0
    return len(set(dedupe(retrieved)[:k]) & expected) / len(expected)


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(len(values) * p) - 1))
    return values[idx]
