import os

os.environ["HF_HOME"] = r"B:\huggingface_cache"

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

print("1. Starting")

print("2. Loading Transformers pipeline...")

try:
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-0.6B",
        temperature=0.5
    )
    print("3. Pipeline loaded successfully")

except Exception as e:
    print("PIPELINE ERROR:")
    print(type(e).__name__)
    print(e)
    raise

print("4. Creating HuggingFacePipeline")

llm = HuggingFacePipeline(
    pipeline=pipe
)

print("5. Creating ChatHuggingFace")

model = ChatHuggingFace(llm=llm)

print("6. Invoking model")

try:
    response = model.invoke("Who is the PM of India?")
    print("7. Response:")
    print(response.content)

except Exception as e:
    print("INFERENCE ERROR:")
    print(type(e).__name__)
    print(e)
    raise