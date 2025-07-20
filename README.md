# RAG-Approach: Retrieval Augmented Generation for Accident Reports

This code demonstrates a  Retrieval Augmented Generation (RAG) architecture for analyzing workplace accident reports using OpenAI embeddings and FAISS vector store. It also displays the euclidean distance between the similar reports which helped guide the LLM's decision. The lower the Euclidean distance, the more similar the reports.

---
# How it Works
<img width="810" height="507" alt="Screenshot 2025-07-20 at 1 27 59 PM" src="https://github.com/user-attachments/assets/a7dd4af0-cb88-403f-a233-84c3185c2205" />

When you type in a new accident report, the program goes through the following steps:

1. **Text Embedding:**  
   The report is converted into a numerical format using a tool from OpenAI. This process is called *embedding*, and it turns your report into a high-dimensional vector (think of it like plotting your report as a point in a giant multi-dimensional graph based on meaning, not just words).

2. **Similarity Search with FAISS:**  
   FAISS (Facebook AI Similarity Search) is a system that efficiently searches for reports that are *most similar* to the one you just typed in.  
   It does this by comparing your report's vector to vectors of existing reports using a measurement called **Euclidean distance**. The closer two reports are in this "vector space," the more similar they are in meaning.

3. **Retrieving Top Matches:**  
   The 5 most similar past reports are retrieved. Their similarity to your report is displayed as a numerical distance — **lower = more similar**.

4. **Decision by the Language Model (LLM):**  
   The retrieved reports are shown to a large AI model (LLM) from OpenAI along with your input.  
   The model uses both your report and the similar examples to decide whether the report should be classified as **"Fall"** or **"Not a Fall"**.

5. **Output:**  
   - You receive the prediction (Fall or Not a Fall).  
   - You also see the top 5 similar reports, how close they were, and their classifications — this provides us further transparency into how the AI made its decision.

> *Note: Program currently does not reference OSHA standards/regulations.*




## Setup Instructions

Follow these steps to set up and run the code locally.

### 1. Clone the Repository

```bash
git clone https://github.com/malakamahdy/RAG-Approach
cd RAG-Approach
```

### 2. Create and Activate a Virtual Environment

It is *highly* recommended to use a virtual environment to manage dependencies.

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```
You’ll know it’s active when you see (venv) at the start of your terminal prompt.

To deactivate it when you're done using the program:
```bash
deactivate
```
> *Note: No need to navigate into the `venv` folder — just activate from your project root folder, `RAG-Approach`.*

### 3. Install Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Set up the `.env` File

In the root folder of the project, `RAG-Approach`, create a new file named:

```
.env
```
> *Note: Yes, the file does start with a dot!*

Open it in any text editor (VS Code, Notepad, etc.).

Add the following line:
```
OPENAI_API_KEY=openai_api_key_here
```

Replace `openai_api_key_here` with the actual OpenAI API key.

### 5. Prepare the Data

Make sure the file `accident_examples.csv` is in the project directory. This file contains the accident reports used for retrieval.

### 6. Run the Program

Run the main script to start the interactive RAG system:

```bash
python rag.py
```

You will be prompted to enter accident reports. Type your input and get predictions and similar report examples.

To exit the program, type `exit` or `quit`.

---

**Developed by Malak Mahdy — mmahdy@islander.tamucc.edu**

---
