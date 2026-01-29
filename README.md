# Engineering Docs RAG

A production-style **Retrieval-Augmented Generation (RAG)** system
for querying large engineering documentation using **semantic search**,  
**grounded LLM generation**, and **evidence-based confidence scoring**.

The project is designed to simulate an **internal enterprise documentation assistant**
with a strong emphasis on correctness, transparency, and system design.


---

## Features
- Ingests large PDF engineering documentation
- Cleans and chunks text with page-level metadata
- Offline embedding and indexing into Qdrant
- Semantic vector search at query time
- Gemini powered answer generation
- Deterministic grounding validation to reject hallucinations
- Evidence-based confidence scoring for answers
- Secure API key management via environment variables

---

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/DeadBoTt-exe/DocuMind-AI.git
```

### Create and activate a virtual environment
```bash
python -m venv .venv
```

Windows
```bash
.venv\Scripts\activate
```

macOS / Linux
```bash
source .venv/bin/activate
```

### Install dependencies

```python
pip install -r requirements.txt
```

### Configure environment variables

Create a .env file in the project root:
```bash
GEMINI_API_KEY=your_api_key_here
```
---

## Running the System

Start Qdrant (Docker) 

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
- If you dont have Docker, Set it up seperately.


Index Documentation:
```python
python -m code.index_documents
```
This step:
- Parses the PDF
- Cleans and chunks text
- Generates embeddings
- Stores vectors in Qdrant

Run the API:
```bash
python -m code.main
```
It will be here - 
```bash
http://127.0.0.1:8000
```
Interactive docs will be here:
```bash
http://127.0.0.1:8000/docs
```

Exampele Query - 
```
{
  "question": "What is a service control policy in AWS Organizations?"
}
```
---


## Tech Stack

- LLM: Gemini
- Vector DB: Qdrant
- Embeddings: Sentence Transformers 
- API: FastAPI
- Validation: Custom logic with selective LangChain usage

---

## Project Structure

```
DocuMind
|
├── code/
│   ├── __init__.py
│   ├── chunker.py
│   ├── cleaner.py
│   ├── confidence.py
│   ├── embeddings.py
│   ├── index_documents.py
│   ├── ingest.py
│   ├── validator.py
│   ├── rag.py
│   └── main.py  
│
├── docs/
│   ├── AWS SDKs and Tools - Reference Guide - aws-sdk-ref.pdf
│   ├── organizations-userguide.pdf
│  
├── .gitignore
├── .env
├── LICENSE
├── requirements.txt
└── README.md

```

