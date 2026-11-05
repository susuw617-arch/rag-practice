import json
import os
from pathlib import Path

# 使用电脑中已经下载好的模型
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer


# 1. 从本地文件加载向量和文本块
project_dir = Path(__file__).resolve().parent
index_dir = project_dir / "index"
vectors_file = index_dir / "vectors.npy"
chunks_file = index_dir / "chunks.json"

if not vectors_file.exists() or not chunks_file.exists():
    raise FileNotFoundError("没有找到本地向量，请先运行 build_vectors.py")

vectors = np.load(vectors_file)

with open(chunks_file, encoding="utf-8") as f:
    chunks = json.load(f)

# 2. 加载模型，模型只负责转换用户的新问题
model = SentenceTransformer("BAAI/bge-m3")


# 3. 让用户输入问题
question = input("请输入你的问题：")

# 4. 把问题也转换成向量
question_vector = model.encode(
    question,
    normalize_embeddings=True,
)

# 5. 比较问题和所有文本块的相似程度
scores = vectors @ question_vector

# 6. 设置需要返回几个文本块
top_k = 1

# 7. 找到相似度最高的前K个文本块
top_indices = scores.argsort()[::-1][:top_k]

# 8. 依次显示检索结果
for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[index]

    print("=" * 50)
    print("排名：", rank)
    print("来源：", chunk["source"])
    print("块编号：", chunk["chunk_id"])
    print("相似度：", scores[index])
    print("内容：")
    print(chunk["text"])
