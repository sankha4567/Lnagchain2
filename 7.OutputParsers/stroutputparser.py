from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import os
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


model=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="groq/compound-mini",
    
  
    temperature=0
)

template1=PromptTemplate(
  template="write a detailed report on {topic}",
  input_variables=["topic"]


)


template2=PromptTemplate(
  template="Summarize the following text clearly and concisely with in 300 words:\n\n{text}.",
  input_variables=["text"]


)
parser=StrOutputParser()
chain1 = template1 | model | parser | template2 | model | parser
response1=chain1.invoke({
  "topic":" Constipation"
})


print(" summraized version ..............................")
print(response1)