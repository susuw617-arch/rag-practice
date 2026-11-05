from chunk import split_text
import os
os.environ["HF_HUB_OFFLINE"] = "1"   # 离线模式，直接用本地的，不联网

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

with open("data/03_家庭光伏系统使用手册.txt", encoding="utf-8") as f:
    text = f.read()
chunk_list = split_text(text)

vectors = model.encode(chunk_list, normalize_embeddings=True)

print(vectors.shape)

