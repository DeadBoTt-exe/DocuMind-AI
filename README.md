# DocuMind — Engineering Docs RAG

---

A production-style **Retrieval-Augmented Generation (RAG)** system
for querying large engineering documentation using **semantic search**,  
**grounded LLM generation**, and **evidence-based confidence scoring**.

The project is designed to simulate an **internal enterprise documentation assistant**
with a strong emphasis on correctness, transparency, and system design.

---

## Features
- Ingests large PDF engineering documentation with layout-aware chunking
- Cleans and chunks text with page-level metadata and deterministic UUID identity
- Offline embedding and indexing into Qdrant (local or Qdrant Cloud)
- Semantic vector search at query time (`all-mpnet-base-v2`)
- Gemini powered answer generation (`gemini-2.5-flash`)
- LLM-based grounding validation (Gemini, temperature 0) that rejects answers not supported by retrieved context; fails closed (answer withheld, `validation.skipped: true`) if the validator is unavailable
- Evidence-based confidence scoring combining retrieval score and chunk coverage
- Runtime observability with in-process query latency, error counters, and circuit breaker metrics
- Optional API key authentication (`X-API-Key`) protecting `/ask` against unauthorized quota spend
- Pre-baked Docker container image ready for Google Cloud Run deployment
- Automated evaluation harness with a 20-question benchmark dataset

---

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/DeadBoTt-exe/DocuMind-AI.git
cd DocuMind-AI
```

### Create and activate a virtual environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

To install development and testing tools:
```bash
pip install -e ".[dev]"
```

Download the spaCy sentence tokenizer model:
```bash
python -m spacy download en_core_web_sm
```

### Configure environment variables

Create a `.env` file in the project root (see `.env.example`):
```bash
GEMINI_API_KEY=your_api_key_here

# Optional: Hosted Qdrant (e.g. Qdrant Cloud). Leave unset for localhost:6333:
# QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
# QDRANT_API_KEY=your_qdrant_key

# Optional: Secure the /ask endpoint with an API key
# API_KEY=your_secret_api_key_here
```

---

## Running the System

### 1. Start Qdrant (Local Docker)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
*(Or use a managed cluster on [Qdrant Cloud](https://cloud.qdrant.io) — see [docs/qdrant-cloud-setup.md](docs/qdrant-cloud-setup.md)).*

### 2. Index Documentation
```bash
python -m documind.index_documents
```
This step:
- Parses the PDF documentation in `docs/`
- Cleans boilerplate, table fragments, and headers
- Chunks text along sentence boundaries with stable UUIDs
- Generates 768-dimensional embeddings via Sentence Transformers
- Creates the collection (if missing) and upserts vectors into Qdrant

### 3. Run the API
```bash
uvicorn documind.main:app --host 127.0.0.1 --port 8000
```

- API Base: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs (Swagger): `http://127.0.0.1:8000/docs`
- Health Liveness: `GET /health`
- Qdrant Keep-Alive: `GET /health/qdrant` (Reads collection point count without calling Gemini)
- Runtime Metrics: `GET /metrics` (Query count, average latency, circuit breaker status)

### Example Query
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a service control policy in AWS Organizations?"}'
```

---

## Evaluation Benchmark

DocuMind includes an automated benchmarking suite to measure retrieval recall, grounding accuracy, and hallucination rates:

```bash
# Run benchmark in mock mode (CI/testing without spending Gemini quota):
python -m evals.evaluate --mock

# Run benchmark against live Qdrant and Gemini engine:
python -m evals.evaluate
```

The benchmark tests 20 curated scenarios:
- **10 Directly Answerable** queries on core AWS Organizations architecture.
- **5 Tricky / Boundary** queries on limits, quotas, and structural invariants.
- **5 Out-of-Domain Negative Controls** (Azure, Kubernetes, ML, trivia) where the model must cleanly refuse to answer (`"I don't know based on the documentation."`) without hallucinating.

---

## Deployment

DocuMind is containerized and production-ready for **Google Cloud Run**:
- Memory allocation: `2Gi` (accommodates PyTorch and pre-cached SentenceTransformer weights).
- Zero cold-start download timeout: Model weights baked into `/opt/hf`.
- Quota protection: `--max-instances 2` cap and `API_KEY` header verification.

See detailed setup in [docs/cloud-run-deployment.md](docs/cloud-run-deployment.md) and [docs/qdrant-cloud-setup.md](docs/qdrant-cloud-setup.md).

---

## Tech Stack

- **LLM**: Google Gemini (`gemini-2.5-flash`) via `google-genai`
- **Vector DB**: Qdrant (`qdrant-client`)
- **Embeddings**: Sentence Transformers (`all-mpnet-base-v2`)
- **Web API**: FastAPI + Uvicorn
- **Validation**: Gemini-based grounding check (temperature 0) with circuit breaker
- **Benchmarking**: Rich CLI reporting & JSON export

---

## Project Structure

```
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
│   ├── organizations-userguide.pdf
│   ├── cloud-run-deployment.md
│   └── qdrant-cloud-setup.md
│
├── evals/
│   ├── __init__.py
│   ├── baseline_results.json
│   ├── benchmark_questions.json
│   └── evaluate.py
│
├── tests/
│   ├── api/
│   │   └── test_api.py
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_cleaner.py
│   │   ├── test_confidence.py
│   │   ├── test_config_qdrant.py
│   │   ├── test_evals.py
│   │   ├── test_hardening_fixes.py
│   │   ├── test_logging_config.py
│   │   ├── test_question_model.py
│   │   ├── test_rag_engine.py
│   │   └── test_validator.py
│   └── conftest.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```
