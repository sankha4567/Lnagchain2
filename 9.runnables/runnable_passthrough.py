from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough
import os
load_dotenv()
prompt=PromptTemplate(
  template="generate a joke on this topic {topic}",
  input_variables=["topic"]
)
model=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="groq/compound-mini",
    
  
    temperature=0
)


parser=StrOutputParser()
prompt2=PromptTemplate(
  template="Explain the following joke-{text}",
  input_variables=["text"]
)
joke_gen_chain=RunnableSequence(prompt,model,parser)
parallel_chain=RunnableParallel(
  {
    "joke":RunnablePassthrough(),
    "explanation":RunnableSequence(prompt2,model,parser)
  }
)
final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
result=final_chain.invoke({
  'topic':"cricket"

})
print(result)
