from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

url = "https://example.com"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)

prompt = PromptTemplate(
    template="""
Answer the following question based only on the given text.

Question:
{question}

Text:
{text}
""",
    input_variables=["question", "text"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({
    "question": "What is this page about?",
    "text": docs[0].page_content
})

print(response)