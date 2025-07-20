import os
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


# Load env vars
load_dotenv()

# Load CSV
df = pd.read_csv("accident_examples.csv")

# Convert CSV rows to Documents
docs = [
    Document(page_content=row["text"], metadata={"label": row["label"]})
    for _, row in df.iterrows()
]

# Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Create FAISS vectorstore
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", k=5)

# Setup RAG QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

print("Type 'exit' or 'quit' to stop.")

while True:
    user_report = input("\nEnter a new accident report: ")
    if user_report.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = qa.invoke({

        "query": f"""
You are analyzing workplace accident reports. 
Given this new report, classify it as "Fall" or "Not a Fall".
Respond with only one word: Fall or Not a Fall.

New Report:
\"\"\"
{user_report}
\"\"\"
"""
    })

    print("\nPrediction:", response["result"].strip())
    print("\nRetrieved Examples:")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\nExample {i+1}:")
        print(f"Text: {doc.page_content.strip()}")
        print(f"Label: {doc.metadata.get('label')}")
