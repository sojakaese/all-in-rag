from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import TextLoader

# 先初始化两个不同的文本分割器
markdown_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
)
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# 1. 加载Markdown文档,这里是一篇开源的小说
loader = TextLoader("../../data/C2/md/Ninety-Three.md", encoding="utf-8")
markdown_text = loader.load()[0].page_content

# 2. 先按Markdown标题拆分，得到大块文本列表，这个列表内并非文本，而是一列Document对象，封装了文本和相应的元数据。
chunks_by_markdown_header = markdown_header_splitter.split_text(markdown_text)

# 3. 对每个含有元数据的大块，再用RecursiveCharacterTextSplitter做细粒度拆分，结果即为先按标题再按字符递归细分的文本块。
final_chunks = recursive_splitter.split_documents(chunks_by_markdown_header)

# 这里预览最终结果的前五个块
preview_number: int = 5
print(
    f"Final_chunks has {len(final_chunks)} chunks:\n ---\nFirst {preview_number} chunks:\n ---"
)

(
    lambda page_number: [
        print("Chunk " + str(page + 1) + ":\n" + final_chunks[page].page_content + "\n")
        for page in range(page_number)
    ]
)(preview_number)
