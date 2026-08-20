from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    task="conversational",
    max_new_tokens=5000,
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Model Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

template=load_prompt("template.json")
if st.button("Summarize"):
  chain=template | model
  result=chain.invoke({
   "paper_input":paper_input,
   "style_input":style_input,
   "length_input":length_input
  })
  print(paper_input)
  print(style_input)
  print(length_input)
  print(result)
  st.write(result.content)