"""Elasticsearch 客户端与索引管理。

文档索引同时承载：
1. BM25 关键词检索（title/content 全文索引）
2. 向量语义检索（embedding: dense_vector + int8_hnsw）
3. 长期记忆（enterprise_memory 索引）
"""

import time

from elasticsearch import Elasticsearch, helpers

import config


def get_es():
    """创建 ES 客户端（ES 9.x，本地开发关闭了安全认证）。"""
    return Elasticsearch(
        config.ES_URL,
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


def doc_mapping(dims: int, shards: int = 1) -> dict:
    """文档索引 mapping：BM25 字段 + 1024 维余弦向量。"""
    return {
        "settings": {
            "number_of_shards": shards,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
            "translog": {"durability": "async", "sync_interval": "30s"},
        },
        "mappings": {
            "properties": {
                "uid": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "content": {"type": "text", "analyzer": "standard"},
                "content_len": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 100},
                },
            }
        },
    }


def memory_mapping(dims: int) -> dict:
    """长期记忆索引 mapping。"""
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "session_id": {"type": "keyword"},
                "role": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "ts": {"type": "date"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "int8_hnsw"},
                },
            }
        },
    }


def create_doc_index(es: Elasticsearch, recreate: bool = False, shards: int = 1):
    """创建文档索引；recreate=True 时先删除重建。"""
    name = config.ES_DOC_INDEX
    if recreate and es.indices.exists(index=name):
        es.indices.delete(index=name)
        print(f"[ES] 已删除旧索引 {name}")
    if not es.indices.exists(index=name):
        es.indices.create(index=name, body=doc_mapping(config.EMBED_DIM, shards))
        print(f"[ES] 已创建索引 {name}（{shards} shard，{config.EMBED_DIM} 维）")


def ensure_memory_index(es: Elasticsearch):
    """长期记忆索引按需创建。"""
    name = config.ES_MEMORY_INDEX
    if not es.indices.exists(index=name):
        es.indices.create(index=name, body=memory_mapping(config.EMBED_DIM))


def bm25_search(es: Elasticsearch, query: str, size: int = 50, source_types=None):
    """BM25 检索：title 权重 3 倍，content 全文匹配。"""
    must = [{
        "multi_match": {
            "query": query,
            "fields": ["title^3", "content"],
            "type": "best_fields",
            "operator": "or",
        }
    }]
    filters = []
    if source_types:
        filters.append({"terms": {"source_type": list(source_types)}})

    body = {
        "size": size,
        "query": {"bool": {"must": must, "filter": filters}},
        "_source": {"excludes": ["embedding"]},
    }
    return es.search(index=config.ES_DOC_INDEX, body=body)["hits"]["hits"]


def vector_search(es: Elasticsearch, vector, k: int = 50, num_candidates: int = 100,
                  source_types=None):
    """kNN 向量检索。"""
    knn = {
        "field": "embedding",
        "query_vector": list(map(float, vector)),
        "k": k,
        "num_candidates": max(num_candidates, k * 2),
    }
    if source_types:
        knn["filter"] = {"terms": {"source_type": list(source_types)}}
    body = {
        "size": k,
        "knn": knn,
        "_source": {"excludes": ["embedding"]},
    }
    return es.search(index=config.ES_DOC_INDEX, body=body)["hits"]["hits"]


def count_docs(es: Elasticsearch, with_vectors: bool = False) -> int:
    query = {"exists": {"field": "embedding"}} if with_vectors else {"match_all": {}}
    return es.count(index=config.ES_DOC_INDEX, query=query)["count"]


def refresh(es: Elasticsearch):
    es.indices.refresh(index=config.ES_DOC_INDEX)


def wait_ready(es: Elasticsearch, timeout: int = 120):
    """等待 ES 集群 green/yellow。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = es.cluster.health(wait_for_status="yellow", timeout="5s")
            if health["status"] in ("green", "yellow"):
                return health
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"ES 在 {timeout}s 内未就绪: {config.ES_URL}")


def bulk_index(es: Elasticsearch, actions, chunk_size: int = 500):
    """helpers.bulk 的薄封装，返回成功条数。"""
    ok = 0
    for success, info in helpers.streaming_bulk(
        es, actions, chunk_size=chunk_size, max_chunk_bytes=50 * 1024 * 1024,
        raise_on_error=False, request_timeout=120,
    ):
        if success:
            ok += 1
        elif info:
            print(f"[ES] 写入失败: {str(info)[:200]}")
    return ok
