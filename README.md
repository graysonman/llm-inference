# Baseline LLM Inference Server

FastAPI service for local LLM inference with a staged `v1` API surface for chat, evaluation, embeddings, RAG, datasets, batch evals, and observability.

## Implemented API surface

- `GET /health`, `GET /healthz`, `GET /readyz`, `GET /v1/health`
- `POST /chat`, `POST /v1/chat`
- `POST /evaluate`, `POST /v1/evaluate`
- `POST /v1/embeddings`
- `POST /v1/rag`, `POST /v1/rag/query`
- `POST /v1/datasets`, `POST /v1/datasets/upload`
- `GET /v1/datasets`, `GET /v1/datasets/{dataset_id}`
- `POST /v1/batch-evals`
- `GET /v1/batch-evals/{run_id}`
- `GET /v1/batch-evals/{run_id}/result`
- `GET /v1/batch-evals/{run_id}/distribution`
- `GET /v1/batch-evals/{run_id}/failures`
- `GET /v1/evals/{run_id}`
- `GET /v1/rag/indexes/{index_id}`
- `GET /v1/metrics`, `GET /metrics/dashboard`

All protected endpoints require `x-api-key` (default local key: `dev-local-key`).

Role/scope auth can be configured with:
- `API_KEY_ROLE` for the default `API_KEY` role (default: `admin`)
- `API_KEYS_JSON` for per-key role/scope overrides (JSON object)

Persistence backend can be configured with:
- `STATE_BACKEND=json|sqlite` (default: `json`)
- `STATE_FILE_PATH` for JSON snapshot backend
- `STATE_SQLITE_PATH` for SQLite backend

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\activate
```

## Example request

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "x-api-key: dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain transformers in 3 sentences.","max_new_tokens":120}'
```

## OpenAPI artifact export

Export the current OpenAPI schema for client generation/contract checks:

```bash
python scripts/export_openapi.py --output docs/openapi.v1.json
```

## Contract verification tests

```bash
python -m pytest -q
```

## Current known gaps vs MVP docs

- Persistence is file-backed JSON state (not yet a production database).
- Redis cache is integrated for chat hot path when `REDIS_URL` is set; service falls back to in-memory cache when Redis is unavailable.
- Batch evaluation now runs asynchronously in-process; external queue/retry orchestration is still pending.
