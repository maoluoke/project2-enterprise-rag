"""分级记忆：Redis 短期记忆（最近若干轮）+ Elasticsearch 长期记忆（跨会话）。

短期记忆只保留最近 ST_MAX_MESSAGES 条消息并设置 TTL，控制 prompt 长度；
长期记忆把每轮问答写入 ES 并向量化，检索时只取与当前问题最相关的少量片段，
避免把整段历史塞进 prompt。
"""

import json
from datetime import datetime, timezone

import redis
from elasticsearch import Elasticsearch

import config
import es_client
from embedding import embed_documents, embed_query


class ShortTermMemory:
    """Redis List：每个 session 一个 key，保留最近 N 条消息。"""

    def __init__(self, client: redis.Redis):
        self.client = client

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"rag:st:{user_id}:{session_id}"

    def add(self, user_id: str, session_id: str, role: str, content: str):
        key = self._key(user_id, session_id)
        item = json.dumps({
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False)
        pipe = self.client.pipeline()
        pipe.rpush(key, item)
        pipe.ltrim(key, -config.ST_MAX_MESSAGES, -1)
        pipe.expire(key, config.ST_TTL_SECONDS)
        pipe.execute()

    def recent(self, user_id: str, session_id: str, limit: int = None):
        limit = limit or config.ST_MAX_MESSAGES
        raw = self.client.lrange(self._key(user_id, session_id), -limit, -1)
        return [json.loads(x) for x in raw]

    def clear(self, user_id: str, session_id: str):
        self.client.delete(self._key(user_id, session_id))


class LongTermMemory:
    """ES 长期记忆：内容 + 1024 维向量，按用户隔离检索。"""

    def __init__(self, es: Elasticsearch):
        self.es = es
        es_client.ensure_memory_index(es)

    def add(self, user_id: str, session_id: str, role: str, content: str, vector=None):
        vector = vector if vector is not None else embed_documents([content])[0]
        self.es.index(
            index=config.ES_MEMORY_INDEX,
            document={
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "content": content[:2000],
                "ts": datetime.now(timezone.utc).isoformat(),
                "embedding": list(map(float, vector)),
            },
        )

    def search(self, user_id: str, query: str, k: int = None):
        k = k or config.LT_TOP_K
        qv = embed_query(query)
        body = {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": list(map(float, qv)),
                "k": k,
                "num_candidates": max(50, k * 10),
                "filter": {"term": {"user_id": user_id}},
            },
            "_source": {"excludes": ["embedding"]},
        }
        hits = self.es.search(index=config.ES_MEMORY_INDEX, body=body)["hits"]["hits"]
        return [dict(h["_source"], score=float(h["_score"])) for h in hits]


class MemoryManager:
    """对外统一接口：写入一轮问答、构建用于 prompt 的记忆上下文。"""

    def __init__(self, user_id: str = "default", session_id: str = "default"):
        self.user_id = user_id
        self.session_id = session_id
        self.es = es_client.get_es()
        try:
            client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
            client.ping()
        except Exception as e:
            raise RuntimeError(
                f"无法连接 Redis {config.REDIS_URL}。请先启动项目内 Redis：\n"
                f"LD_LIBRARY_PATH={config.BASE_DIR}/tools/redis/usr/lib/x86_64-linux-gnu "
                f"{config.BASE_DIR}/tools/redis/usr/bin/redis-server --port 6379 --daemonize yes"
            ) from e
        self.short = ShortTermMemory(client)
        self.long = LongTermMemory(self.es)

    def add_turn(self, user_text: str, assistant_text: str):
        self.short.add(self.user_id, self.session_id, "user", user_text)
        self.short.add(self.user_id, self.session_id, "assistant", assistant_text)
        vectors = embed_documents([user_text, assistant_text])
        self.long.add(self.user_id, self.session_id, "user", user_text, vectors[0])
        self.long.add(self.user_id, self.session_id, "assistant", assistant_text, vectors[1])

    def build_context(self, query: str, recent_limit: int = 6, long_limit: int = 2):
        """返回 (近期消息列表, 长期记忆文本, 长期记忆命中列表)。"""
        recent = self.short.recent(self.user_id, self.session_id, limit=recent_limit * 2)
        messages = [
            {"role": m["role"], "content": m["content"][:500]} for m in recent
        ]

        long_hits = []
        try:
            long_hits = self.long.search(self.user_id, query, k=long_limit)
        except Exception as e:
            print(f"[Memory] 长期记忆检索失败，跳过: {e}")

        recent_contents = {m["content"][:100] for m in messages}
        long_hits = [h for h in long_hits if h["content"][:100] not in recent_contents]

        if long_hits:
            long_text = "\n".join(
                f"- [{h['ts'][:19]}] {h['role']}: {h['content'][:300]}" for h in long_hits
            )
        else:
            long_text = "（无）"
        return messages, long_text, long_hits

    def clear(self):
        self.short.clear(self.user_id, self.session_id)
