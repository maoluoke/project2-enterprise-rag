"""Step 2：构建 Elasticsearch 混合索引（BM25 + 向量）。

用法：
    python 02_build_es_index.py --recreate          # 重建并写入样例
    python 02_build_es_index.py --recreate --skip-vectors
    python 02_build_es_index.py --vectors-only      # 断点续跑向量阶段
"""

from indexer import main

if __name__ == "__main__":
    main()
