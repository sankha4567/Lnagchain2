# Paragraph 1: History of cricket
# Paragraph 2: History of cricket
# Paragraph 3: History of cricket

#                 ↓

# Chunk 1: Cricket history


# Paragraph 4: Cricket rules
# Paragraph 5: Cricket rules

#                 ↓

# Chunk 2: Cricket rules


# Paragraph 6: Cricket World Cup
# Paragraph 7: Cricket World Cup

#                 ↓

# Chunk 3: Cricket tournaments

from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("cricket.txt", encoding="utf-8")

documents = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

splitter = SemanticChunker(
    embeddings
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Semantic Chunk {i + 1} ---")
    print(chunk.page_content)