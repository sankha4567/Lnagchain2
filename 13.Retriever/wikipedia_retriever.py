from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(
  top_k_results=3,
  lang="en"
)
query="What is the greatest moment of cricket"
docs=retriever.invoke(query)
for i,doc in enumerate(docs):
  print(f"\n---- Result {i+1} ----")
  print(f"Content:\n{doc.page_content}----")