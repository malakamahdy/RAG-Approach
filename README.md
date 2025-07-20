# RAG-Approach: Retrieval Augmented Generation for Accident Reports

This code demonstrates a  Retrieval Augmented Generation (RAG) architecture for analyzing workplace accident reports using OpenAI embeddings and FAISS vector store. It also displays the euclidean distance between the similar reports which helped guide the LLM's decision. The lower euclidean distance, the more similar the reports.

---
<img width="810" height="507" alt="Screenshot 2025-07-20 at 1 27 59 PM" src="https://github.com/user-attachments/assets/a7dd4af0-cb88-403f-a233-84c3185c2205" />


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

### 3. Install Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Set up the `.env` File

Create a `.env` file in the root directory of the project to store your OpenAI API key:

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

## Notes

- Ensure your API key has permissions to access OpenAI Embeddings and GPT-4o.
- The vector store uses FAISS for efficient similarity search.
- Lower similarity scores indicate more semantically similar documents.

---

## Troubleshooting

- If you encounter line-ending warnings with Git (`CRLF will be replaced by LF`), it can be safely ignored.
- Do not commit your `venv/` directory to the repository.
- If you have any questions, feel free to contact the maintainer.

---

**Developed by Malak Mahdy — mmahdy@islander.tamucc.edu**

---
