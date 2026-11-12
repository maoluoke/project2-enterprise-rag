# 项目2：企业级混合检索知识库系统

面向企业非结构化文档（Slack / Gmail / Google Drive / Linear / GitHub 等）的精准检索与问答系统。基于 **EnterpriseRAG-Bench** 数据集构建，支持十万级文档的毫秒级混合检索，并提供无向量库的轻量化降级方案。

## 核心特性

| 特性 | 说明 | 实测效果（2万篇样例，500道Golden问题） |
|---|---|---|
| 混合检索 | BM25 关键词 + 向量语义加权融合（或 RRF），BGE-Reranker 精排 | HitRate@5 0.781，较纯 BM25 **+27.4%** |
| 轻量 DocParser | File System + Hash 倒排索引，不依赖向量数据库 | 2万篇索引 148MB / 构建 12s / 检索 p50=31ms |
| 分级记忆 | Redis 短期记忆（滚动窗口+TTL）+ ES 向量长期记忆 | 支持指代改写与跨会话上下文召回 |
| 自动化评估 | Golden Dataset 消融评估，HitRate / MRR / Recall / 延迟 | 4 模式对比报告 JSON+CSV 落盘，可上报 LangFuse |
| 智能体能力 | Function Calling 工具决策 + MCP 联网/对外服务 + Skill 按需加载 | 内部问题自动调知识库，外部实时问题自动联网并标注来源 |

## 技术栈

- **检索引擎**：Elasticsearch 9.5（BM25 + int8_hnsw 向量索引，单索引双能力）
- **嵌入模型**：Qwen3-Embedding-4B（本地，MRL 截断 1024 维）
- **生成模型**：Qwen3-8B（本地 bf16）或 DashScope 兼容接口（可切换）
- **精排模型**：bge-reranker-base（交叉编码器，长文档分窗打分）
- **记忆/缓存**：Redis 6（短期）、ES（长期）
- **编排**：LangChain（Prompt 链 + Query 改写）

## 系统架构

```
                    ┌──────────── Redis 短期记忆（最近12条, TTL 1h）
   用户问题 ──┬─────┤
             │      └──────────── ES 长期记忆（向量化，按用户隔离检索 Top-2）
             ▼
      Query 改写（Qwen3-8B，解析指代）
             ▼
   ┌── BM25 召回（title³+content）──┐
   ├────────────────────────────────┤──► 加权融合/RRF（Top-50）
   └── 向量召回（Qwen3-Embedding）──┘
             ▼
      BGE-Reranker 精排（全文分窗取最高分，Top-5）
             ▼
      Qwen3-8B 生成答案（引用 doc_id）──► 写回两级记忆
```

## 目录结构

```
project2/
├── 01_prepare_data.py     # 数据准备：parquet → JSONL，强制包含全部Golden标准文档
├── 02_build_es_index.py   # ES索引构建：阶段1全量BM25，阶段2向量（断点续跑）
├── 03_hybrid_search.py    # 混合检索CLI（4种模式对比）
├── 04_rag_answer.py       # RAG问答（LangChain + 分级记忆）
├── 05_docparser_lite.py   # 轻量方案CLI：ingest / ask / benchmark
├── 06_evaluate.py         # 自动化评估流水线
├── 07_agent.py            # 智能体：Function Calling + MCP联网 + Skill，自主决策调用工具
├── config.py              # 全局配置（均可环境变量覆盖）
├── es_client.py           # ES客户端与索引管理
├── embedding.py           # Qwen3-Embedding-4B 封装
├── indexer.py             # 索引构建核心：BM25阶段 + 向量阶段（checkpoint断点续跑）
├── reranker.py            # BGE精排（长文档分窗策略）
├── retrieval.py           # 混合检索核心（融合算法）
├── memory.py              # 分级记忆管理
├── llm.py                 # 生成模型封装（本地/DashScope）
├── docparser.py           # 轻量化检索核心：FileSystem + Hash倒排索引 + BM25
├── eval_utils.py          # HitRate / MRR / Recall 指标计算
├── skill_loader.py        # Skill 装载器：技能索引注入 + 按需加载完整指令
├── kb_tools.py            # kb_search 工具封装（qwen-agent 装饰器参考实现）
├── kb_mcp_server.py       # 自定义 MCP Server：对外暴露 kb_search / kb_stats
├── skills/                # 技能库（kb-research / web-fact-check / weekly-report）
├── docstore/              # DocParser 文档存储（按哈希分目录的 JSON）
├── data/                  # docs.jsonl / golden.jsonl / 评估报告 / 向量checkpoint
└── tools/redis/           # 绿色版 Redis 二进制（免安装）
```

## 快速开始

### 0. 前置条件

- Python：`/home/zkx456/other/bin/python`（已装好 requirements.txt 全部依赖）
- GPU：RTX 5090 D 32GB（嵌入+精排+8B生成约 25GB 显存；无 GPU 可设 `EMBED_DEVICE=cpu` + `LLM_BACKEND=dashscope`）

### 1. 启动 ES 和 Redis

```bash
# ES（注意：数据/日志目录必须在纯ASCII路径下，中文路径会导致 AccessDeniedException）
ES_JAVA_OPTS="-Xms4g -Xmx4g" \
/home/zkx456/lsks/elasticsearch-9.5.4-linux-x86_64/elasticsearch-9.5.4/bin/elasticsearch \
  -E http.port=9201 -E xpack.security.enabled=false \
  -E discovery.type=single-node \
  -E path.data=/home/zkx456/lsks/es_project2_data \
  -E path.logs=/home/zkx456/lsks/es_project2_logs \
  -E cluster.name=project2 &

# Redis（项目内绿色版，需指定库路径）
export LD_LIBRARY_PATH="/media/zkx456/系统/ai课件/project2/tools/redis/usr/lib/x86_64-linux-gnu"
"/media/zkx456/系统/ai课件/project2/tools/redis/usr/bin/redis-server" \
  --port 6379 --daemonize yes \
  --dir "/media/zkx456/系统/ai课件/project2/redis_data"
```

### 2. 准备数据并建索引

```bash
cd /media/zkx456/系统/ai课件/project2

# 2万篇样例（强制含全部723篇Golden标准文档）+ 500道Golden问题
python 01_prepare_data.py --sample 20000

# 建索引：先BM25（~1分钟），后向量（~8分钟，支持断点续跑）
python 02_build_es_index.py --recreate          # 完整构建
python 02_build_es_index.py --vectors-only      # 中断后继续向量阶段
```

### 3. 使用

```bash
# 混合检索（4种模式：bm25 / vector / hybrid / hybrid_rerank）
python 03_hybrid_search.py --mode hybrid_rerank "What caused the 5xx surge?"
python 03_hybrid_search.py                      # 交互模式

# RAG问答（带分级记忆，同 session 多轮自动改写指代）
python 04_rag_answer.py --user alice --session s1 "What is the default max_file_size?"
python 04_rag_answer.py --user alice --session s1 "Can it be adjusted per tenant?"
python 04_rag_answer.py --no-memory "..."       # 纯单轮

# 轻量方案（无需向量库）
python 05_docparser_lite.py --ingest 20000      # 构建
python 05_docparser_lite.py --ask "..."         # 问答
python 05_docparser_lite.py --benchmark 100     # 延迟/命中率测试

# 自动化评估（4模式消融）
python 06_evaluate.py --limit 0                 # 全部500题
python 06_evaluate.py --limit 50 --modes bm25,hybrid_rerank
python 06_evaluate.py --langfuse                # 上报LangFuse（需配环境变量）

# 智能体（Function Calling + MCP 联网搜索 + Skill）
python 07_agent.py "公司 multipart 上传限制？"   # 内部问题 -> 自动调 kb_search
python 07_agent.py "今天美元汇率？"             # 外部问题 -> 自动调 Tavily 联网
python 07_agent.py "汇总 gate5 进展生成简报"    # 匹配技能 -> 先 load_skill 再执行
python 07_agent.py                             # 交互模式

# 自定义 MCP Server（把知识库能力暴露给其他 Agent 客户端）
python kb_mcp_server.py                        # 提供 kb_search / kb_stats 两个工具
```

### Skill 机制

`skills/` 目录下每个子目录是一个技能（SKILL.md：front matter 声明 name/description，正文为指令）。启动时只把技能索引注入 system prompt；模型判断用户意图匹配某技能时，先调用 `load_skill` 拉取完整指令，再按指令执行——避免所有指令常驻 prompt 浪费 token。

| 技能 | 触发场景 |
|---|---|
| kb-research | 复杂内部问题：拆解子问题 → 多跳检索 → 交叉验证 |
| web-fact-check | 内部答案与公开信息核对（时效词/价格/汇率） |
| weekly-report | 周报/进展汇总：广度检索 → 结构化中文简报 |

## 评估结果（全量 500 道 Golden 问题）

| 模式 | HitRate@5 | MRR@5 | Recall@5 | p50 | p95 |
|---|---|---|---|---|---|
| bm25 | 0.613 | 0.491 | 0.554 | 5.7ms | 7.1ms |
| vector | 0.655 | 0.570 | 0.597 | 17.4ms | 19.1ms |
| hybrid | 0.781 | 0.686 | 0.735 | 23.1ms | 25.5ms |
| hybrid_rerank | 0.781 | 0.695 | 0.726 | 142ms | 165ms |

结论：融合检索在几乎不增加延迟的情况下将 HitRate 从 0.61 提升到 0.78；精排主要提升排序质量（MRR），适合对 Top-1 精度敏感的场景。

## 关键设计

**长文档精排**：文档平均 4800 字符远超 reranker 的 512 token 上限，直接截断会丢失尾部答案。`reranker.py` 将全文按 500 字符窗口（重叠 350）均匀切窗，逐窗打分取最高分，并返回命中窗口作为生成上下文。

**两阶段索引**：BM25 先行让系统快速可用；向量阶段逐批（32篇/批）生成并 update 写入，`data/vector_checkpoint.json` 记录进度，中断后 `--vectors-only` 续跑不重复消耗 GPU。

**融合算法**：默认加权融合（两路分数按各自最高分归一化，`alpha=0.5`），`--rrf` 切换为倒数排名融合，避免分数量纲不一致。

**查询改写**：多轮对话时用 Qwen3-8B 将指代/省略改写为独立检索查询（如 "Can it be adjusted?" → "Can the size limits be adjusted?"）。

## 常用配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `HYBRID_ALPHA` | 0.5 | 向量权重（BM25=1-alpha） |
| `RECALL_K` / `FINAL_K` | 50 / 5 | 召回数 / 精排后输出数 |
| `LLM_BACKEND` | local | `dashscope` 切云端生成 |
| `EMBED_DEVICE` | cuda | 无显卡设 `cpu` |
| `MIN_RERANK_SCORE` | 0 | 精排分数阈值（用于无答案拒识） |

## 扩展到全量 51 万篇

```bash
python 01_prepare_data.py --sample 0
python 02_build_es_index.py --recreate    # 向量阶段约 3.3 小时
```

## 已知注意事项

- **import 顺序**：任何脚本必须先 `import embedding`（sentence_transformers）再 `import torch`，否则本机 pyarrow 与 torch 存在 libstdc++ CXXABI 冲突。
- **ES 路径**：数据/日志目录不能含中文（JVM 会抛 AccessDeniedException），本机使用 `/home/zkx456/lsks/es_project2_*`。
- **ES 端口**：使用 9201，与已有的 9200 实例隔离，互不影响。
- **数据集协议**：EnterpriseRAG-Bench 用于技术验证与评估，商用前请确认其许可条款。
