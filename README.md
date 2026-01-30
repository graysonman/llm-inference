# Baseline LLM Inference Server (Stage 1)

A minimal transformer-backed inference service. This is the foundational “Stage 1” of a larger project that will evolve into evaluation, RAG, agents, and dashboards later.

## What this is
- FastAPI server exposing a `/chat` endpoint backed by a Hugging Face transformer model
- Basic observability:
  - latency (ms)
  - prompt token count
  - completion token count
- Dockerized and reproducible

## What this is NOT (yet)
- No RAG
- No embeddings endpoint
- No evaluation datasets / dashboards
- No Redis caching
- No authentication / API keys
- No agents

Those come in later stages.

## Why does this exist?
To expose the operational realities of running transformer models before adding higher-level reasoning or evaluation.
---

## Requirements
- Python 3.11+ (recommended)
- (Optional) Docker

By default, the server uses `distilgpt2` to keep setup lightweight.

---

## Quickstart (Local)

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .\.venv\Scripts\activate # windows powershell
```

### 2) Install Requirements
```bash
pip install -r requirements.txt
```

### 3) Run
```bash
export MODEL_NAME=distilgpt2   # mac/linux
# set MODEL_NAME=distilgpt2    # windows cmd
uvicorn app.main:app --reload --port 8000
```

### 4) Test
```bash
curl http://localhost:8000/health
```

### 5) Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain transformers simply in 3 sentences.","max_new_tokens":120,"temperature":0.2,"top_p":0.95}'
```

## Quickstart (Docker)

### 1) Create a docker environment
```bash
docker build -t llm-server .
```

### 2) Run
```bash
docker run -p 8000:8000 -e MODEL_NAME=distilgpt2 llm-server
```

