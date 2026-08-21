from langchain_core.runnables import (
    RunnableBranch,
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough
)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on this topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a concise summary of the following report:\n\n{text}",
    input_variables=["text"]
)

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = StrOutputParser()

report_generation_chain = RunnableSequence(
    prompt1,
    model,
    parser
)

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300,
        RunnableSequence(
            RunnableLambda(lambda x: {"text": x}),
            prompt2,
            model,
            parser
        )
    ),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_generation_chain,
    branch_chain
)

response = final_chain.invoke({
    "topic": "cricket"
})

print(response)