"""项目2 全局配置：路径、ES/Redis地址、模型与检索参数。

所有配置均可用环境变量覆盖，方便在样例/全量、不同机器之间切换。
运行环境：/home/zkx456/other/bin/python
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOC_STORE_DIR = BASE_DIR / "docstore"
ES_DATA_DIR = Path(os.getenv("ES_DATA_DIR", str(BASE_DIR / "es_data")))

# --- 原始数据（EnterpriseRAG-Bench）---
RAW_DOCS_PATH = Path(os.getenv(
    "ERB_DOCS_PARQUET",
    "/media/zkx456/系统/ai课件/EnterpriseRAG-Bench/data/documents/test.parquet",
))
RAW_QUESTIONS_PATH = Path(os.getenv(
    "ERB_QUESTIONS_PARQUET",
    "/media/zkx456/系统/ai课件/EnterpriseRAG-Bench/data/questions/test.parquet",
))

# --- Elasticsearch / Redis ---
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9201")
ES_DOC_INDEX = os.getenv("ES_DOC_INDEX", "enterprise_docs")
ES_MEMORY_INDEX = os.getenv("ES_MEMORY_INDEX", "enterprise_memory")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# --- 嵌入模型：Qwen3-Embedding-4B，MRL 截断到 1024 维 ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "512"))   # 文档侧最大 token 数
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "1200"))  # 送入向量的正文字符上限
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))

# --- 精排模型：bge-reranker-base 交叉编码器 ---
RERANK_MODEL_DIR = os.getenv(
    "RERANK_MODEL_DIR",
    "/media/zkx456/系统/ai课件/11-RAG调优/10-RAG高级技术与调优/CASE-高效召回/models/BAAI/bge-reranker-base",
)
RERANK_WINDOW = int(os.getenv("RERANK_WINDOW", "500"))   # 文档切窗长度（字符）
RERANK_STRIDE = int(os.getenv("RERANK_STRIDE", "350"))   # 窗口重叠步长
RERANK_MAX_PASSAGES = int(os.getenv("RERANK_MAX_PASSAGES", "6"))

# --- 生成模型：Qwen3-8B（本地）或 DashScope OpenAI 兼容接口 ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")          # local | dashscope
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
LLM_LOAD_IN_4BIT = os.getenv("LLM_LOAD_IN_4BIT", "0") == "1"

# --- 检索参数 ---
RECALL_K = int(os.getenv("RECALL_K", "50"))       # 单路召回/融合候选数
FINAL_K = int(os.getenv("FINAL_K", "5"))          # 精排后进入生成的文档数
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # 向量权重，BM25 权重 = 1-alpha
FUSION_METHOD = os.getenv("FUSION_METHOD", "weighted")  # weighted | rrf
RRF_K = int(os.getenv("RRF_K", "60"))
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))

# --- 记忆参数 ---
ST_TTL_SECONDS = int(os.getenv("ST_TTL_SECONDS", "3600"))
ST_MAX_MESSAGES = int(os.getenv("ST_MAX_MESSAGES", "12"))
LT_TOP_K = int(os.getenv("LT_TOP_K", "3"))

# --- 数据准备 ---
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20000"))   # 0 表示全量 51 万篇
GOLDEN_SIZE = int(os.getenv("GOLDEN_SIZE", "0"))      # 0 表示使用全部 500 题
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# 启动时统一离线加载本地模型，避免联网校验
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_dirs():
    """创建项目内数据目录。"""
    for d in (DATA_DIR, DOC_STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)
