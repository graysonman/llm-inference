# Baseline LLM Inference Server (Stage 1 + v1 Contract Skeleton)

A minimal transformer-backed inference service. This is the foundational “Stage 1” of a larger project that will evolve into evaluation, RAG, agents, and dashboards later.

## What this is
- FastAPI server exposing a `/chat` endpoint backed by a Hugging Face transformer model
- Basic observability:
  - latency (ms)
  - prompt token count
  - completion token count
- v1 contract skeleton endpoints (`/healthz`, `/readyz`, `/v1/*`) for upcoming staged implementation
- Dockerized and reproducible

## What this is NOT (yet)
- No RAG
- No embeddings endpoint
- No evaluation datasets / dashboards
- No Redis caching
- No authentication / API keys
- No agents

Those come in later stages. The `/v1/*` endpoints are now present as contract-valid placeholders to support incremental implementation.

## API key for v1 endpoints

All `/v1/*` endpoints require `x-api-key`.

- Default local key: `dev-local-key`
- Override with env var: `API_KEY`

Example:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "x-api-key: dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'
```

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
curl http://localhost:8000/healthz
```

### 5) Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "x-api-key: dev-local-key" \
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


## Metrics

`GET /v1/metrics` returns baseline chat/cache/rate-limit counters for milestone tracking.

```bash
# JSON view
curl http://localhost:8000/v1/metrics \
  -H "x-api-key: dev-local-key"

# Prometheus/OpenMetrics-style plaintext view
curl "http://localhost:8000/v1/metrics?format=prometheus" \
  -H "x-api-key: dev-local-key"
```


## Persistence-backed stubs

The v1 skeleton now keeps in-memory records for datasets and evaluation runs so you can exercise read paths while durable storage is being implemented.

- `GET /v1/evals/{run_id}` returns a stored single-sample evaluation run.
- `GET /v1/batch-evals/{run_id}` returns batch run status/progress metadata.
- `GET /v1/rag/indexes/{index_id}` returns basic index status for uploaded datasets.


## Rate limiting

Protected endpoints are rate-limited per API key (default `120` requests/minute, configurable with `RATE_LIMIT_PER_MINUTE`).

On `429` responses the API returns canonical error body details and these headers:

- `Retry-After`
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`


## Dataset + batch eval contract coverage

Newly implemented contract-aligned endpoints:

- `GET /v1/datasets` (supports `type`, `status`, `limit`)
- `GET /v1/datasets/{dataset_id}`
- `GET /v1/batch-evals/{run_id}/result`
- `GET /v1/batch-evals/{run_id}/distribution`
- `GET /v1/batch-evals/{run_id}/failures`

`POST /v1/datasets` now accepts `type` and `metadata` and returns `status` + `created_at`.
`POST /v1/batch-evals` now returns `202 Accepted` and stores in-memory summary/failure/distribution artifacts for read endpoints.


`GET /v1/health` is available as a versioned liveness/readiness alias with model/device metadata.
