# LLM Inference Console

FastAPI-based LLM operations platform with:
- Chat and evaluation APIs
- RAG + dataset management
- Batch evaluation orchestration
- Authz (role + scope) with capability-aware UI
- Admin operations (runtime config, maintenance, breaker, SLO, runbooks, audit/state)
- Optional vector backend swap (`in_memory`/`faiss`)
- Optional tracing backend integration (OTLP)
- Two web surfaces:
  - Main console: `http://localhost:8000`
  - Targeted React surface: `http://localhost:8000/react`

## What This Project Solves

This project gives you one local service/UI to:
- Test prompts and RAG behavior quickly
- Upload datasets and run repeatable evaluations
- Track quality and operational health in one place
- Operate the service safely via admin controls
- Verify auth scopes/capabilities per API key

## Core Features

- Inference: `chat`, `evaluate`, `embeddings`, `rag`
- Data plane: dataset upload/list/get/delete/restore/purge
- Eval plane: batch queue, status, retries, events stream, artifact export
- Ops plane: metrics, readiness/health, maintenance mode, circuit breaker, runtime tuning profiles, SLO incidents, runbooks/templates
- Security: API key auth + role/scope enforcement + capability discovery endpoints
- Stretch features: vector backend status, tracing status/probe, agent tools endpoints, targeted React migration surface

## Prerequisites

- Python 3.11 (for local run)
- Docker Desktop (for container run)

## Quick Start (Docker, recommended)

1. From repo root, create `.env`:

```powershell
Copy-Item .env.example .env
```

2. Build and run:

```powershell
docker compose up -d --build
```

3. Open:
- Main UI: `http://localhost:8000`
- React UI: `http://localhost:8000/react`

4. In UI auth box:
- API key: `dev-local-key`
- Click `Use Key`

5. Stop:

```powershell
docker compose down
```

## Quick Start (Local Python)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Minimal `.env` Configuration

```env
API_KEY=dev-local-key
API_KEY_ROLE=admin
API_KEYS_JSON={"dev-local-key":{"role":"admin","scopes":["*"]}}
MODEL_NAME=distilgpt2

STATE_PERSISTENCE_ENABLED=1
STATE_BACKEND=json
```

Common optional config:
- `STATE_BACKEND=sqlite`
- `VECTOR_INDEX_BACKEND=in_memory|faiss`
- `TRACING_ENABLED=1`
- `TRACING_OTLP_ENDPOINT=http://<collector>:4318/v1/traces`

![Main UI](/readme-img/image.png)

## How To Use The UI

1. `Playground`: run chat inference and inspect latency/tokens.
2. `RAG Explorer`: query a chosen `dataset_id` with retrieval context.
3. `Dataset Manager`: upload `.jsonl/.csv/.txt`, list datasets.
4. `Batch Evaluation`: run async dataset evals and inspect result.
5. `Metrics Dashboard`: summary + Prometheus text.
6. `Agent Tools`: discover allowed tools and run tool-based plans.
7. `System Status`: health/readiness/model/auth/access matrix/rag backend/tracing.
8. `Admin Ops` (admin role): runtime config/profiles, maintenance, breaker, SLO incidents, runbooks, audit/state operations.

## API Examples

Set variables:

```bash
export BASE_URL="http://localhost:8000"
export API_KEY="dev-local-key"
```

Chat:

```bash
curl -s -X POST "$BASE_URL/v1/chat" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain transformers in 3 sentences.","max_new_tokens":120}'
```

Upload dataset:

```bash
curl -s -X POST "$BASE_URL/v1/datasets/upload" \
  -H "x-api-key: $API_KEY" \
  -F "name=quick-rag" \
  -F "type=rag_corpus" \
  -F "file=@./quick_rag.jsonl"
```

RAG query:

```bash
curl -s -X POST "$BASE_URL/v1/rag/query" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"ds_xxxxxxxx","query":"How should API keys be rotated?","top_k":5}'
```

Agent tools:

```bash
curl -s "$BASE_URL/v1/agent/tools" -H "x-api-key: $API_KEY"

curl -s -X POST "$BASE_URL/v1/agent/run" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"goal":"Investigate datasets and metrics","requested_tools":["datasets.list","metrics.dashboard"]}'
```

Vector/tracing status:

```bash
curl -s "$BASE_URL/v1/rag/vector-backend" -H "x-api-key: $API_KEY"
curl -s "$BASE_URL/v1/tracing/status" -H "x-api-key: $API_KEY"
```

## Troubleshooting

- Tabs/buttons disabled except Status:
  - Check `GET /v1/auth/context` with your key.
  - Ensure `.env` has `API_KEYS_JSON` or role/scopes that allow required capabilities.
  - Rebuild container after env changes: `docker compose up -d --build`.
  - Hard refresh browser (`Ctrl+F5`) after frontend changes.

- `docker compose down` says no configuration file:
  - Run command from repo root where `docker-compose.yml` exists.

- Startup takes time:
  - First run downloads model weights; wait for `Application startup complete`.

## Dev Utilities

Run tests:

```powershell
$env:PYTHONPATH='.'
pytest -q
```

Export OpenAPI:

```powershell
python scripts/export_openapi.py --output docs/openapi.v1.json
```
