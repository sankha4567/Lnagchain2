from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import os
load_dotenv()


model=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    max_tokens=4000,
  
    temperature=0
)

template1=PromptTemplate(
  template="write a detailed report on {topic}",
  input_variables=["topic"]


)
template2=PromptTemplate(
  template="Summarize the following text clearly and concisely:\n\n{text}.",
  input_variables=["text"]


)
prompt1=template1.invoke({"topic":"black hole"})


response=model.invoke(prompt1)
print(response.content)
prompt2=template2.invoke({
  "text":response.content
})
total_response=model.invoke(prompt2)
print(total_response.content)
