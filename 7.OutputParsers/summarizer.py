from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import os
load_dotenv()

os.environ["HF_HOME"] = r"B:\huggingface_cache"
pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-0.6B",
        temperature=0.5,
        max_new_tokens=600,
        return_full_text=False
    )
llm=HuggingFacePipeline(
  pipeline=pipe
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

model=ChatHuggingFace(llm=llm)
response=model.invoke(prompt1)
prompt2=template2.invoke({
  "text":response.content
})
total_response=model.invoke(prompt2)
print(total_response.content)
