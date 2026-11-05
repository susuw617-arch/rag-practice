from loader import load_documents
from chunk import split_text


def build_chunks():
    # 读取14篇完整文章
    documents = load_documents()

    # 用来存放所有文章的文本块
    all_chunks = []

    for document in documents:
        # 切分当前文章
        chunk_texts = split_text(document["text"])

        for chunk_id, chunk_text in enumerate(chunk_texts):
            chunk = {
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": chunk_text,
            }

            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()

    print(f"一共生成了 {len(chunks)} 个文本块")

    # 展示前3个文本块
    for chunk in chunks[:3]:
        print("=" * 50)
        print("来源：", chunk["source"])
        print("块编号：", chunk["chunk_id"])
        print("块长度：", len(chunk["text"]))
        print("内容：", chunk["text"][:100])