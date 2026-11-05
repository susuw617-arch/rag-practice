from build_vectors import chunks, vectors, model


# 1. 让用户输入问题
question = input("请输入你的问题：")

# 2. 把问题也转换成向量
question_vector = model.encode(
    question,
    normalize_embeddings=True,
)

# 3. 比较问题和所有文本块的相似程度
scores = vectors @ question_vector

# 4. 设置需要返回几个文本块
top_k = 1

# 5. 找到相似度最高的前3个文本块
top_indices = scores.argsort()[::-1][:top_k]

# 6. 依次显示检索结果
for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[index]

    print("=" * 50)
    print("排名：", rank)
    print("来源：", chunk["source"])
    print("块编号：", chunk["chunk_id"])
    print("相似度：", scores[index])
    print("内容：")
    print(chunk["text"])