from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import TextLoader

# 定义文本分割器
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
)
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# 1. 文档加载
loader = TextLoader("../../data/C2/md/Ninety-Three.md", encoding="utf-8")
markdown_text = loader.load()[0].page_content

# 2. 先按标题拆分，得到大块文本列表
chunks_by_markdown_header = header_splitter.split_text(markdown_text)

# 3. 对每个大块，再用RecursiveCharacterTextSplitter做细粒度拆分
final_chunks = []
for chunk in chunks_by_markdown_header:
    small_chunks = recursive_splitter.split_text(chunk.page_content)
    final_chunks.extend(small_chunks)

# final_chunks中即为先按标题再按字符递归细分的文本块,这里预览前五个块
preview_number: int = 5
print(
    f"Final_chunks has {len(final_chunks)} chunks:\n ---\nFirst {preview_number} chunks:\n ---\n"
)

(
    lambda page_number: [
        print("Chunk " + str(page) + ":\n" + final_chunks[page] + "\n\n")
        for page in range(page_number)
    ]
)(preview_number)
