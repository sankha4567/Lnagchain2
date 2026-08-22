from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("resume.pdf")

documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separator="\n"
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)


# 2. Recursive length-based splitting

# This is the more commonly used general-purpose splitter.


loader = PyPDFLoader("resume.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)