
##structuredoutputparser enforce llm to give response in terms of the schema / predefined format but does not enforce llm to do return data with proper data validaton 
from langchain_core.output_parsers.structured import StructuredOutputParser , ResponseSchema
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



schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)