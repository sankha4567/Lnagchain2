from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

prompt = PromptTemplate(
    template="Write a summary of the following text:\n\n{text}",
    input_variables=["text"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "text": docs[0].page_content
})

print(result)