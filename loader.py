from pathlib import Path


def load_documents(data_dir="data"):
    documents = []

    # 找到 data 文件夹中所有 txt 文件
    file_paths = sorted(Path(data_dir).glob("*.txt"))

    for file_path in file_paths:
        # 读取文章内容
        text = file_path.read_text(encoding="utf-8")

        # 同时保存文章内容和文件名
        document = {
            "source": file_path.name,
            "text": text,
        }

        documents.append(document)

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"一共读取了 {len(documents)} 篇文章")

    for document in documents:
        print(
            document["source"],
            "文章长度：",
            len(document["text"]),
        )