from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

doc1 = Document(
    page_content="Virat Kohli is the most successful and consistent batsman in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="MS Dhoni is one of the most successful captains in IPL and has led Chennai Super Kings.",
    metadata={"team": "Chennai Super Kings"}
)

doc3 = Document(
    page_content="Rohit Sharma has captained Mumbai Indians to several IPL championships.",
    metadata={"team": "Mumbai Indians"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is regarded as one of the best fast bowlers in IPL history.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Andre Russell is a powerful all-rounder known for his explosive batting.",
    metadata={"team": "Kolkata Knight Riders"}
)

doc6 = Document(
    page_content="Ravindra Jadeja is an important all-rounder for Chennai Super Kings.",
    metadata={"team": "Chennai Super Kings"}
)

doc7 = Document(
    page_content="Hardik Pandya is known for his all-round performances and leadership.",
    metadata={"team": "Mumbai Indians"}
)

doc8 = Document(
    page_content="Rashid Khan is one of the most successful spinners in IPL.",
    metadata={"team": "Gujarat Titans"}
)

doc9 = Document(
    page_content="Shubman Gill played a crucial role in leading Gujarat Titans.",
    metadata={"team": "Gujarat Titans"}
)

doc10 = Document(
    page_content="KL Rahul is known for his elegant batting technique and consistency.",
    metadata={"team": "Lucknow Super Giants"}
)

documents = [
    doc1,
    doc2,
    doc3,
    doc4,
    doc5,
    doc6,
    doc7,
    doc8,
    doc9,
    doc10
]

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="simple",
    persist_directory="my_chroma_db"
)

vector_store.add_documents(documents=documents)

print(
    vector_store.get(
        include=["embeddings", "documents", "metadatas"]
    )
)

ans = vector_store.similarity_search(
    query="who is the best batsman in IPL",
    k=3
)

for document in ans:
    print(document)

filter_ans = vector_store.similarity_search_with_score(
    query="best player",
    filter={"team": "Mumbai Indians"}
)

print(filter_ans)

print(
    vector_store.delete(
        ids=["79be2e9-733c-48e1-a57b-c4c007526eb1"]
    )
)