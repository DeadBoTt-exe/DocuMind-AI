# Engineering Docs RAG

---

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
- LLM-based grounding validation (Gemini, temperature 0) that rejects answers not supported by the retrieved context; fails closed (answer withheld, `validation.skipped: true`) if the validator is unavailable
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
# Optional - use a hosted Qdrant (e.g. Qdrant Cloud) instead of localhost:
# QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
# QDRANT_API_KEY=your_qdrant_key
# (see .env.example; keys belong in .env, never in code)
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
python -m documind.index_documents
```
This step:
- Parses the PDF
- Cleans and chunks text
- Generates embeddings
- Stores vectors in Qdrant

Run the API:
```bash
uvicorn documind.main:app --host 127.0.0.1 --port 8000
```
It will be here - 
```bash
http://127.0.0.1:8000
```
Keep-alive for hosted (free) Qdrant clusters, which are suspended after a week idle:
`GET /health/qdrant` reads from Qdrant without calling Gemini. Point a scheduler or uptime monitor at it.
If a cluster is ever deleted, re-run `python -m documind.index_documents` to rebuild the index.

Interactive docs will be here:
```bash
http://127.0.0.1:8000/docs
```

Example Query - 
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
- Validation: Gemini-based grounding check with a circuit breaker

---

## Project Structure

`
DocuMind
|
├── documind/
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── metrics_route.py
│   │   └── questions.py 
│   ├── __init__.py
│   ├── chunker.py
│   ├── cleaner.py
│   ├── confidence.py
│   ├── config.py
│   ├── embeddings.py
│   ├── index_documents.py
│   ├── ingest.py
│   ├── logging_config.py
│   ├── main.py
│   ├── metrics.py
│   ├── prompts.py
│   ├── rag.py
│   └── validator.py
│
├── docs/
│   ├── AWS SDKs and Tools - Reference Guide - aws-sdk-ref.pdf
│   └── organizations-userguide.pdf
│
├── tests/
│   ├── api/
│   │   └── test_api.py
│   └── unit/
│       ├── test_chunker.py
│       ├── test_cleaner.py
│       ├── test_confidence.py
│       ├── test_config_qdrant.py
│       ├── test_logging_config.py
│       ├── test_question_model.py
│       ├── test_rag_engine.py
│       └── test_validator.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
`
