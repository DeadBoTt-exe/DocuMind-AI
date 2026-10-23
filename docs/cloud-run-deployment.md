# Cloud Run Deployment Guide for DocuMind RAG

This document outlines the recommended configuration and commands to deploy DocuMind to Google Cloud Run.

---

## 1. Prerequisites & Secrets Setup

Store API keys in **Google Secret Manager** rather than passing raw environment variables:

```bash
# 1. Create secrets in Secret Manager
echo -n "YOUR_GEMINI_KEY" | gcloud secrets create documind-gemini-key --data-file=-
echo -n "YOUR_QDRANT_API_KEY" | gcloud secrets create documind-qdrant-key --data-file=-
echo -n "YOUR_APP_API_KEY" | gcloud secrets create documind-app-key --data-file=-

# 2. Grant Secret Accessor role to Cloud Run default service account
PROJECT_NUM=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding documind-gemini-key \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding documind-qdrant-key \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding documind-app-key \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 2. One-Time Vector Indexing (Qdrant Cloud)

Before your first deployment, your Qdrant Cloud cluster must be provisioned and indexed:

```bash
# Set your remote Qdrant credentials
export QDRANT_URL="https://<your-cluster-id>.<region>.gcp.cloud.qdrant.io:6333"
export QDRANT_API_KEY="<your-qdrant-key>"

# Run indexing
python -m documind.index_documents
```

---

## 3. Container Build & Push

Build the container image using Google Cloud Build and Artifact Registry (Container Registry `gcr.io` is deprecated):

```bash
# 1. Create Artifact Registry Docker repository (if not already created)
gcloud artifacts repositories create documind-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repository for DocuMind RAG"

# 2. Build via Cloud Build and push to Artifact Registry
PROJECT_ID=$(gcloud config get-value project)
gcloud builds submit --tag us-central1-docker.pkg.dev/${PROJECT_ID}/documind-repo/documind:latest .
```

---

## 4. Cloud Run Service Deployment

Deploy the container with production-grade resource allocation, startup CPU boost, and quota caps:

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud run deploy documind \
  --image us-central1-docker.pkg.dev/${PROJECT_ID}/documind-repo/documind:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 10 \
  --timeout 60s \
  --set-env-vars QDRANT_URL="https://<your-cluster-id>.<region>.gcp.cloud.qdrant.io:6333",LOG_LEVEL="INFO" \
  --set-secrets GEMINI_API_KEY=documind-gemini-key:latest,QDRANT_API_KEY=documind-qdrant-key:latest,API_KEY=documind-app-key:latest
```

### Key Deployment Parameters Explained:
- `--cpu-boost`: Temporarily provides 100% additional CPU during container boot, substantially reducing cold-start times when PyTorch and `all-mpnet-base-v2` load into memory.
- `--memory 2Gi`: Required for SentenceTransformer (`all-mpnet-base-v2`) in-memory weights and PyTorch CPU inference.
- `--max-instances 2`: Strict safety cap preventing runaway instance scaling and guarding Gemini API quota against unexpected spikes.
- `--timeout 60s`: Allows ample headroom for RAG generation and grounding validation without prematurely terminating queries.
- `--set-secrets API_KEY=...`: Protects the `/ask` endpoint so only clients presenting the `X-API-Key` header can trigger queries.
- Startup probe: Cloud Run automatically pings the container port on startup; the embedded model weights in `/opt/hf` allow fast cold starts (< 5s).

---

## 5. Automated Cluster Keep-Alive

Free Qdrant Cloud clusters are suspended after 7 days of inactivity. Point a free uptime monitor or Cloud Scheduler job at the keep-alive route every 2 days:

```bash
# Cloud Scheduler keep-alive job
gcloud scheduler jobs create http documind-qdrant-ping \
  --schedule="0 0 */2 * *" \
  --uri="https://<YOUR-CLOUD-RUN-URL>/health/qdrant" \
  --http-method=GET
```
This endpoint executes a lightweight point-count read from Qdrant without incurring any Gemini LLM costs.
