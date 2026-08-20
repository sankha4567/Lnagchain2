import os

os.environ["HF_HOME"] = r"B:\huggingface_cache"

from transformers import pipeline

print("Starting model load...")

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

print("MODEL LOADED!")

messages = [
    {"role": "user", "content": "What is the capital of India?"}
]

result = pipe(messages, max_new_tokens=30)

print(result)