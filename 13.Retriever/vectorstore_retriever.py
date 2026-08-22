from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


documents=[
  Document(page_content="Langchain helps developer build LLM applications"),
   Document(page_content="Chroma is a vector database optimized for LLM based search"),
   Document(page_content="Embeddings convert text into high dimensional vectors"),
   Document(page_content="Gemini provides powerful embedding models."),
]
vectorstore=Chroma.from_documents(
  documents=documents,
  embedding=embeddings_model,
  collection_name='my_collection'

)
retriever=vectorstore.as_retriever(
  search_kwargs={
    "k":2
  }
)
result=retriever.invoke(
  "Emdeddings"
)

for i ,res in enumerate(result):
  print(f"\n---- Result {i+1} ----")
  print(f"Content:\n{res.page_content}----")