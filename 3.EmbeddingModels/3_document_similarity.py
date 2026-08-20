from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=300)
documents=[
  "Virat Kohli, India, Batsman",
  "Sachin Tendulkar, India, Batsman",
  "MS Dhoni, India, Wicketkeeper-Batsman",
  "Rohit Sharma, India, Batsman",
  "Jasprit Bumrah, India, Bowler",
  "Kane Williamson, New Zealand, Batsman",
  "Joe Root, England, Batsman",
  "Ben Stokes, England, All-rounder",
  "Babar Azam, Pakistan, Batsman",
  "Pat Cummins, Australia, Bowler"
]
query="tell me about virat kohli"
doc_embeddings=embeddings.embed_documents(documents)
query_embedding=embeddings.embed_query(query)
print(cosine_similarity([query_embedding],doc_embeddings))