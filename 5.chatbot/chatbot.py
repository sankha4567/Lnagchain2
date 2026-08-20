from langchain_core.messages import AIMessage, HumanMessage

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
model=ChatGroq(model="openai/gpt-oss-120b",temperature=0.5, api_key=os.getenv("GROQ_API_KEY"))
chat_history=[]
while True:
   user_input = input("You: ")
   chat_history.append(HumanMessage(content=user_input))
   if user_input == "exit":
      break
   result = model.invoke(chat_history)

   print("AI: ",result.content)
   chat_history.append(AIMessage(content=result.content))