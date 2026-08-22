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


documents = [
    Document(
        page_content="Pollution is the introduction of harmful substances into the environment."
    ),

    Document(
        page_content="The main causes of pollution include vehicle emissions, industrial waste, plastic disposal, and the burning of fossil fuels."
    ),

    Document(
        page_content="Air pollution can cause breathing problems, lung diseases, and other serious health issues."
    ),

    Document(
        page_content="Water pollution harms fish and other aquatic organisms and can contaminate drinking water."
    ),

    Document(
        page_content="Soil pollution can reduce soil fertility and negatively affect agriculture and food production."
    ),

    Document(
        page_content="Pollution can be reduced through recycling, renewable energy, public transportation, and proper waste management."
    ),

    Document(
        page_content="Pollution is a major environmental problem that affects humans, animals, plants, and ecosystems."
    ),

    Document(
        page_content="The government can control pollution by creating environmental laws and regulating industrial emissions."
    ),
]
vectorstore=Chroma.from_documents(
  documents=documents,
  embedding=embeddings_model,
 

)
retriever=vectorstore.as_retriever(
  search_type='mmr',
  search_kwargs={
    "k":2,
    "lambda_mult":0.5
  }
)
result=retriever.invoke(
  "Soil Poullution is what ?"
)

for i ,res in enumerate(result):
  print(f"\n---- Result {i+1} ----")
  print(f"Content:\n{res.page_content}----")