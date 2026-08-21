from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
def remove_thinking(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text
def word_count(text):
  return len(text.split())
prompt=PromptTemplate(
  template="generate a joke on this topic {topic}",
  input_variables=["topic"]
)
model=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    
  
    temperature=0
)


parser=StrOutputParser()
joke_generate_chain=RunnableSequence(prompt,model,parser, RunnableLambda(remove_thinking))
parallel_chain=RunnableParallel({
  "joke":RunnablePassthrough(),
  # "word_count":RunnableLambda(word_count)
  "word_count":RunnableLambda(lambda x:len(x.split()))
})
final_chain=RunnableSequence(joke_generate_chain,parallel_chain)
result=final_chain.invoke({
  "topic":"AI"
})
print(result)
