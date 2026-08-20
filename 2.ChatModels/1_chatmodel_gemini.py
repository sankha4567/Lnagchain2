from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=1.5)
result=model.invoke("write a poem on mamta banerjee")
print(result.content)