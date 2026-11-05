import os

# 使用电脑中已经下载好的模型
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from build_chunks import build_chunks


# 1. 获取106个文本块
chunks = build_chunks()

# 2. 单独取出每个文本块的文字
texts = []

for chunk in chunks:
    texts.append(chunk["text"])

# 3. 加载向量模型
model = SentenceTransformer("BAAI/bge-m3")

# 4. 把106个文本块转换成向量
vectors = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# 5. 查看结果
print("文本块数量：", len(chunks))
print("向量形状：", vectors.shape)