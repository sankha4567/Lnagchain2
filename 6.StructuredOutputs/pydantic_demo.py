from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Literal,Annotated, Optional
from pydantic import BaseModel,Field
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.6-27b",
    temperature=0
)


class Review(BaseModel):
    key_themes:list[str]=Field(description="write down all key themese disscussed in the review in a list")
    summary: str=Field(
      
        description="A brief summary of the review"
    )

    sentiment: Literal["positive","negative","neutral"]=Field(
        "Return sentiment of the review whether positive, negative, or neutral"
    )
    pros:Optional[list[str]]=Field(default=None,description="write down all pros inside a list")
    cons:Optional[list[str]]=Field(default=None,description="write down all cons inside a list")


structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""
I am very disappointed with this product.

While the hardware quality seems decent, the overall user experience is frustrating.

The software is slow, cluttered with unnecessary pre-installed apps, and frequently lags during basic tasks.

The interface feels outdated and unintuitive compared to competing products.

Battery performance is also below expectations, requiring frequent charging throughout the day.

Customer support was unhelpful when I reported these issues.

Overall, the product has potential, but in its current state, I would not recommend it to others.
""")


print(result)
