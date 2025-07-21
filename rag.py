# Retrieval Augmented Generation (RAG) Approach with Chroma
# This code is a basic RAG architecture for testing purposes.
# For information on how to run the code and how it works, review the README file, or email me at mmahdy@islander.tamucc.edu.
# Malak Mahdy

import os
import pandas as pd
from dotenv import load_dotenv

# Required imports
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from colorama import init, Fore, Style
import tempfile

# Load environment variables
load_dotenv()

# Load accident report CSV
df = pd.read_csv("accident_examples.csv")

# Convert CSV rows to Documents
docs = [
    Document(page_content=row["text"], metadata={"label": row["label"]})
    for _, row in df.iterrows()
]

# Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Create embeddings
embedding = OpenAIEmbeddings()

# Create Chroma vectorstore (cross-platform alternative to FAISS)
persist_directory = tempfile.mkdtemp()
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory=persist_directory)

# Create retriever from vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", k=5)

# Setup Retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-4o"),
    retriever=retriever,
    return_source_documents=True
)

# Initialize colorama for colored terminal output
init(autoreset=True)

# Function to get similarity scores
def get_similarity_scores(query, k=5):
    """Get documents with their similarity scores"""
    return vectorstore.similarity_search_with_score(query, k=k)

# Setup prompt
print(Fore.CYAN + Style.BRIGHT + "Type 'exit' or 'quit' to stop.")
print(Fore.YELLOW + "Lower Euclidean (distance) scores = More similar documents")

# Loop for user input
while True:
    print(Fore.MAGENTA + "-" * 50)
    print(Fore.MAGENTA + "-" * 50)
    user_report = input("\nEnter a new accident report: ")
    if user_report.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Get classification from the model
    # This is a temporary prompt which we will refine
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
# ^ Above, it is attaching the above built in prompt, to the user-inputted 
    })

    # Get retrieved examples and similarity scores
    docs_with_scores = get_similarity_scores(user_report, k=5)

    # Print classification
    print(Fore.CYAN + Style.BRIGHT + "\nPrediction:", response["result"].strip())

    # Print retrieved documents with similarity scores
    print("\nRetrieved Examples:")
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\nExample {i+1} (Euclidean Score: {score:.4f}):")
        print(f"Text: {doc.page_content.strip()}")
        print(f"Label: {doc.metadata.get('label')}")

    # Confidence estimation
    scores = [score for _, score in docs_with_scores]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\nAverage similarity: {avg_score:.4f}")
        if avg_score < 0.5:
            print("→ High confidence (very similar examples)")
        elif avg_score < 1.0:
            print("→ Good confidence (similar examples)")
        else:
            print("→ Lower confidence (less similar examples)")
