from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.6-27b",
    temperature=0
)


class Review(TypedDict):
    summary: Annotated[
        str,
        "A brief summary of the review"
    ]

    sentiment: Annotated[
        str,
        "Return sentiment of the review whether positive, negative, or neutral"
    ]


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
