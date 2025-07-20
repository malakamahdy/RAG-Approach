# Retrieval Augmented Generation (RAG) Approach
# This code is a basic RAG architecture for testing purposes.
# For information on how to run the code, check the README file, or email me at mmahdy@islander.tamucc.edu
# Malak Mahdy

import os
import pandas as pd
from dotenv import load_dotenv
# Required imports
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from colorama import init, Fore, Style

# Loading env vars
load_dotenv()

# Loading accident report CSV
df = pd.read_csv("accident_examples.csv")

# Convert CSV rows to Documents
docs = [
    Document(page_content=row["text"], metadata={"label": row["label"]})
    for _, row in df.iterrows()
]

# Splitting documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Creating FAISS vectorstore
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)

# Creating retriever (using the standard retriever)
retriever = vectorstore.as_retriever(search_type="similarity", k=5)

# Setting up RAG QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-4o"),
    retriever=retriever,
    return_source_documents=True
)

# Initialize colorama (just for color coding outputs for easier readability!)
init(autoreset=True)

# Function to get similarity scores separately
# Similarity score is the semantic similarity between the user's inputed report and the retrieved reports from the CSV
def get_similarity_scores(query, k=5):
    """Get documents with their similarity scores"""
    return vectorstore.similarity_search_with_score(query, k=k)

# Setup messages with color
print(Fore.CYAN + Style.BRIGHT + "Type 'exit' or 'quit' to stop.")
print(Fore.YELLOW + "Lower euclidean (distance) scores = More similar documents")

# Loop for handling user continuing or exiting the program
while True:
    print(Fore.MAGENTA + "-" * 50)
    print(Fore.MAGENTA + "-" * 50)
    user_report = input("\nEnter a new accident report: ")
    if user_report.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    # Get the QA response
 
# This is a temporary prompt which we will refine
    response = qa.invoke({
        "query": f"""
You are analyzing workplace accident reports. 
Given this new report, classify it as "Fall" or "Not a Fall".
Respond with only one word: Fall or Not a Fall.
New Report:
\"\"\"
{user_report} # It is attaching the above built in prompt, to the user-inputted 
\"\"\"
"""
    })
    
    # Get similarity scores separately
    docs_with_scores = get_similarity_scores(user_report, k=5)
    
    # Printing prediction
    print(Fore.CYAN + Style.BRIGHT +"\nPrediction:", response["result"].strip())
    
    # Print retrieved examples with similarity scores
    print("\nRetrieved Examples:")
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\nExample {i+1} (Euclidean Score: {score:.4f}):")
        print(f"Text: {doc.page_content.strip()}")
        print(f"Label: {doc.metadata.get('label')}")
    
    # Calculate average similarity for confidence assessment
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