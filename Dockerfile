# Production Docker image for DocuMind RAG (Cloud Run ready)
# Build locally: `docker build -t documind .`
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
WORKDIR /app

# CPU-only torch first (avoids multi-GB CUDA wheels), then the pinned requirements
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Bake the embedding model into the image so cold starts don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

COPY documind ./documind

# Run as a non-root user
RUN useradd --create-home app && chown -R app /opt/hf
USER app

# Cloud Run injects $PORT. Secrets (GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY) come from
# the environment / Secret Manager; there is deliberately no .env in the image.
CMD exec uvicorn documind.main:app --host 0.0.0.0 --port ${PORT:-8080}
