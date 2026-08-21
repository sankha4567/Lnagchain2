from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
prompt=PromptTemplate(
  template="generate a joke on this topic {topic}",
  input_variables=["topic"]
)
model=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.6-27b",
    
  
    temperature=0
)


parser=StrOutputParser()
chain=RunnableSequence(prompt,model,parser)
result=chain.invoke({
  'topic':'cricket'
})
print(result)