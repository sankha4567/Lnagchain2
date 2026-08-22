from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
import os

load_dotenv()


# ---------------- DOCUMENTS ----------------

all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"}
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"}
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"}
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"}
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"}
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"}
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"}
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"}
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global attention.",
        metadata={"source": "I4"}
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"}
    ),
]


# ---------------- EMBEDDINGS ----------------

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


# ---------------- VECTOR STORE ----------------

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings_model
)


# ---------------- BASE RETRIEVER ----------------

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# ---------------- LLM ----------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# ---------------- COMPRESSOR ----------------

compressor = LLMChainExtractor.from_llm(model)


# ---------------- CONTEXTUAL COMPRESSION RETRIEVER ----------------

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


# ---------------- QUERY ----------------

query = "How can I improve my energy levels?"


# ---------------- RETRIEVE ----------------

results = compression_retriever.invoke(query)


# ---------------- OUTPUT ----------------

for i, doc in enumerate(results):
    print(f"\n---- Result {i + 1} ----")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")