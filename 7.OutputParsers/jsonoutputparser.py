from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


import os
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()


model=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="groq/compound-mini",
    
  
    temperature=0
)
parser=JsonOutputParser()
template=PromptTemplate(
  template="Give me five facts about {topic} \n {format_instruction}",
  input_variables=["topic"],
  partial_variables={"format_instruction": parser.get_format_instructions()}
)
chain=template | model | parser
final_result=chain.invoke({
  "topic":"Virat Kohli"
})
print(final_result)
print(type(final_result))