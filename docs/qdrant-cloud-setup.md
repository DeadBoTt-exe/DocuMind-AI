# Qdrant Cloud Provisioning & Indexing Guide

This guide covers setting up a managed Qdrant Cloud cluster, indexing the AWS documentation into it, and configuring keep-alive monitoring.

---

## 1. Create a Free Qdrant Cloud Cluster

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io) and sign up / log in.
2. Click **Create Cluster**:
   - Choose the **Free Tier (1 GB)**.
   - Select cloud provider and region (e.g., `GCP / us-central1` or closest to your deployment).
   - Give the cluster a name (e.g. `documind-cluster`).
3. Under **Data Access Control / API Keys**:
   - Create a new API Key with Read & Write permissions.
   - Copy the **Cluster URL** (e.g. `https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333`).
   - Copy the **API Key**.

---

## 2. Ingest & Index Documentation into Qdrant Cloud

The collection **must exist and be populated** before starting the FastAPI app or deploying to Cloud Run, as `RAGEngine` verifies collection existence on startup.

### Run Indexing Locally Against Qdrant Cloud:

```powershell
# In PowerShell (Windows)
$env:GEMINI_API_KEY="your-gemini-key"
$env:QDRANT_URL="https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333"
$env:QDRANT_API_KEY="your-qdrant-api-key"

python -m documind.index_documents
```

```bash
# In Bash (Linux / macOS)
export GEMINI_API_KEY="your-gemini-key"
export QDRANT_URL="https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your-qdrant-api-key"

python -m documind.index_documents
```

### What this does:
1. Validates connectivity to the remote cluster URL.
2. Parses `docs/organizations-userguide.pdf` (1000+ pages) using PyMuPDF.
3. Cleans AWS boilerplate, headers, and noisy table artifacts.
4. Chunks text along sentence boundaries with deterministic UUIDv5 IDs and metadata (file, page number).
5. Embeds all chunks using Sentence Transformers (`all-mpnet-base-v2`, 768 dimensions).
6. Creates the remote `aws-org-docs` collection with Cosine similarity if it doesn't already exist.
7. Upserts vector points in batches of 64.

---

## 3. Verifying the Index

### Via the Qdrant Cloud Web Console:
- Navigate to your cluster in [cloud.qdrant.io](https://cloud.qdrant.io).
- Click on **Collections** ➔ `aws-org-docs`.
- Confirm that points are populated (typically ~1,300–1,400 chunks for the guide).

### Via the API Keep-Alive Route:
Once the FastAPI server is running with the environment variables set:

```bash
curl http://127.0.0.1:8000/health/qdrant
```

Expected Response:
```json
{
  "status": "ok",
  "qdrant": "ok",
  "points": 1358
}
```

---

## 4. Preventing Inactivity Suspension (Keep-Alive)

Free clusters on Qdrant Cloud are automatically put into hibernation if no queries are received for **7 days**.

The endpoint `GET /health/qdrant` is designed specifically to prevent hibernation:
- It makes a single, lightweight metadata request (`get_collection`).
- It **never** invokes the Gemini LLM or embedding model, costing \$0.00.
- Point any free uptime monitor (UptimeRobot, BetterStack) or a GCP Cloud Scheduler job to ping `GET /health/qdrant` every 48 hours.
