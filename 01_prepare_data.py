"""Step 1：准备文档与 Golden Dataset。

用法：
    python 01_prepare_data.py                 # 默认 2 万篇样例 + 全部 500 题
    python 01_prepare_data.py --sample 0      # 全量 51 万篇
    python 01_prepare_data.py --golden 50     # 分层抽样 50 题
"""

from data_prep import main

if __name__ == "__main__":
    main()
