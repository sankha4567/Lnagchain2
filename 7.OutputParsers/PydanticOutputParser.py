## PydanticOutputParser is a structured output parser in langchain that uses pydantic model to enforce schema validation when processing LLM response.
## it tells llm to return data in terms of well defned structure .Automatically convert LLMS response to python objects.Use pydantic built in validation to catch incorrect or missing data.works well with other Langchain components.
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.6-27b",
    temperature=0
)


class Person(BaseModel):
   name:str=Field(description='name of the person')
   age:int=Field(gt=18,description="age of the person")
   city:str=Field(description="Name of the city person belongs to")
parser=PydanticOutputParser(pydantic_object=Person)
prompt=PromptTemplate(
  template="""
Generate a fictional person from {place}.
Do not return the JSON schema.
Do not explain your answer.
just give the output
{format_instruction}
""",
   input_variables=["place"],
   partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain=prompt | model | parser
response=chain.invoke({
     "place":"Germany"
})
print(response)
