



def split_text(text, chunk_size=300,overlap=50):
    chunks = []
    for start in range(0,len(text),chunk_size - overlap):
        chunk = text[start: start + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == '__main__':
    with open("data/01_城市地下综合管廊说明文档.txt", encoding="utf-8") as f:
        text = f.read()

    chunk_list = split_text(text)
    print(len(chunk_list))

    print(chunk_list[0][-50:])
    print(chunk_list[1][:50])

    for i, c in enumerate(chunk_list):
        print(i, len(c), c[:20])
