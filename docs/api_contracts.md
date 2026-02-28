# API Contracts (v1)

This document defines request/response schemas, error envelopes, and compatibility rules for the inference API surface.

## 1) Versioning and compatibility

- **Base path**: all production endpoints are versioned under `/v1` (for example, `/v1/chat`).
- **Unversioned aliases** (such as `/chat`) may exist during migration, but clients should move to `/v1/*`.
- **Backward-compatible changes** (allowed in `v1`):
  - Add new optional request fields.
  - Add new response fields.
  - Add new enum values only when clients can safely ignore unknown values.
- **Breaking changes** (require `v2`):
  - Remove or rename fields.
  - Change field type/semantics.
  - Make optional fields required.
  - Change error envelope structure.

## 2) Authentication and headers

### Required header
- `x-api-key: <key>` is required for all non-health endpoints.

### Standard request headers
- `Content-Type: application/json` for JSON bodies.
- `x-request-id` (optional): client-provided request correlation ID.

### Standard response headers
- `x-request-id`: request ID used by the server.
- `x-latency-ms`: end-to-end API latency in milliseconds.

## 3) Common response and error envelopes

### Error envelope (all endpoints)

```json
{
  "error": {
    "code": "invalid_request",
    "message": "prompt must be non-empty",
    "details": {
      "field": "prompt"
    }
  },
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

### Error codes (canonical)
- `invalid_request` → HTTP `400`
- `unauthorized` → HTTP `401`
- `forbidden` → HTTP `403`
- `not_found` → HTTP `404`
- `conflict` → HTTP `409`
- `payload_too_large` → HTTP `413`
- `rate_limited` → HTTP `429`
- `internal_error` → HTTP `500`
- `service_unavailable` → HTTP `503`

### Rate-limit error shape (HTTP 429)

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 120,
      "remaining": 0,
      "reset_at": "2026-01-10T15:30:00Z",
      "bucket": "api_key_per_minute"
    }
  },
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

Recommended headers on 429:
- `Retry-After: <seconds>`
- `X-RateLimit-Limit: <int>`
- `X-RateLimit-Remaining: <int>`
- `X-RateLimit-Reset: <unix-epoch-seconds>`

---

## 4) Inference endpoints

## `POST /v1/chat`
Generate a text completion from a prompt.

### Request

```json
{
  "prompt": "Explain transformers simply in 3 sentences.",
  "max_new_tokens": 160,
  "temperature": 0.2,
  "top_p": 0.95,
  "mode": "single",
  "refine_steps": 1,
  "critique_temperature": 0.2
}
```

### Request schema
- `prompt` (string, required, 1..200000 chars)
- `max_new_tokens` (integer, optional, default `160`, range `1..1024`)
- `temperature` (number, optional, default `0.2`, range `0.0..2.0`)
- `top_p` (number, optional, default `0.95`, range `0.0..1.0`)
- `mode` (enum, optional): `single | refine`, default `single`
- `refine_steps` (integer, optional, default `1`, range `1..3`)
- `critique_temperature` (number, optional, default `0.2`, range `0.0..2.0`)

### Success response (200)

```json
{
  "response": "Transformers process tokens in parallel using self-attention...",
  "latency_ms": 187,
  "prompt_tokens": 12,
  "completion_tokens": 64,
  "model": "distilgpt2",
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0",
  "context_window": 1024,
  "context_used_pct": 7.42,
  "model_type": "decoder-only",
  "attention_masking": "causal",
  "attention_heads": 12,
  "hidden_size": 768,
  "estimated_attention_ops": 5776,
  "total_tokens": 76,
  "output_to_input_ratio": 5.3333,
  "refined": false,
  "original_response": null,
  "critique": null,
  "refine_steps_used": 0
}
```

## `POST /v1/evaluate`
Score a model answer against one or more criteria.

### Request

```json
{
  "prompt": "What is overfitting?",
  "response": "Overfitting happens when a model memorizes...",
  "criteria": ["accuracy", "clarity", "overall"]
}
```

### Request schema
- `prompt` (string, required, 1..200000 chars)
- `response` (string, required, 1..200000 chars)
- `criteria` (array of enum, optional, default `["overall"]`)
  - allowed values: `accuracy | clarity | reasoning | factuality | overall`

### Success response (200)

```json
{
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0",
  "model": "distilgpt2",
  "scores": [
    {"criterion": "accuracy", "score": 8, "rationale": "Mostly correct with minor omissions."},
    {"criterion": "clarity", "score": 9, "rationale": "Clear and concise explanation."},
    {"criterion": "overall", "score": 8, "rationale": "Strong answer with slight room to improve precision."}
  ],
  "latency_ms": 242
}
```

## `POST /v1/embeddings`
Return vector embeddings for one or more input texts.

### Request

```json
{
  "input": ["first text", "second text"],
  "model": "bge-small-en-v1.5",
  "normalize": true
}
```

### Request schema
- `input` (string | array[string], required)
- `model` (string, optional; server default if omitted)
- `normalize` (boolean, optional, default `true`)

### Success response (200)

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, -0.441, 0.338]},
    {"object": "embedding", "index": 1, "embedding": [0.209, -0.101, 0.044]}
  ],
  "model": "bge-small-en-v1.5",
  "usage": {"input_tokens": 9, "total_tokens": 9},
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

## `POST /v1/rag`
Run retrieval-augmented generation using a retrieval corpus.

### Request

```json
{
  "query": "How do I rotate API keys safely?",
  "dataset_id": "kb_prod_01",
  "top_k": 5,
  "max_new_tokens": 220,
  "temperature": 0.1
}
```

### Request schema
- `query` (string, required)
- `dataset_id` (string, required)
- `top_k` (integer, optional, default `5`, range `1..50`)
- `max_new_tokens` (integer, optional)
- `temperature` (number, optional)

### Success response (200)

```json
{
  "answer": "Rotate keys by creating a new key, updating clients, then revoking the old key...",
  "citations": [
    {
      "doc_id": "runbook-42",
      "chunk_id": "runbook-42#3",
      "score": 0.89,
      "text": "Key rotation should be phased..."
    }
  ],
  "retrieval": {
    "dataset_id": "kb_prod_01",
    "top_k": 5,
    "latency_ms": 34
  },
  "generation": {
    "model": "distilgpt2",
    "latency_ms": 118
  },
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

---

## 5) Dataset endpoints

## `POST /v1/datasets` (upload)
Upload a dataset for retrieval/evaluation.

### Request (multipart/form-data)
Fields:
- `name` (string, required)
- `type` (enum, required): `rag_corpus | eval_set`
- `file` (file, required): accepted `.jsonl`, `.csv`, `.txt`
- `metadata` (JSON string, optional)

### Success response (201)

```json
{
  "dataset_id": "ds_01J9Q5X0RBMFKC6Q0A9BQZ8NQJ",
  "name": "prod-kb-january",
  "type": "rag_corpus",
  "status": "processing",
  "created_at": "2026-01-10T15:00:00Z",
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

## `GET /v1/datasets` (list)

### Query params
- `type` (optional): `rag_corpus | eval_set`
- `status` (optional): `processing | ready | failed`
- `limit` (optional, default `20`, max `100`)
- `cursor` (optional)

### Success response (200)

```json
{
  "data": [
    {
      "dataset_id": "ds_01J9Q5X0RBMFKC6Q0A9BQZ8NQJ",
      "name": "prod-kb-january",
      "type": "rag_corpus",
      "status": "ready",
      "record_count": 1032,
      "created_at": "2026-01-10T15:00:00Z"
    }
  ],
  "next_cursor": null,
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

## `GET /v1/datasets/{dataset_id}` (get)

### Success response (200)

```json
{
  "dataset_id": "ds_01J9Q5X0RBMFKC6Q0A9BQZ8NQJ",
  "name": "prod-kb-january",
  "type": "rag_corpus",
  "status": "ready",
  "record_count": 1032,
  "error": null,
  "created_at": "2026-01-10T15:00:00Z",
  "updated_at": "2026-01-10T15:02:07Z",
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

---

## 6) Batch evaluation endpoints

## `POST /v1/batch-evals` (create)
Create an asynchronous batch evaluation run.

### Request

```json
{
  "dataset_id": "ds_eval_01",
  "criteria": ["accuracy", "clarity", "overall"],
  "model": "distilgpt2",
  "concurrency": 4
}
```

### Success response (202)

```json
{
  "batch_eval_id": "be_01J9Q6CBJ2A9EDMFGY89AZZ4W5",
  "status": "queued",
  "created_at": "2026-01-10T15:04:00Z",
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

## `GET /v1/batch-evals/{batch_eval_id}` (status)

### Success response (200)

```json
{
  "batch_eval_id": "be_01J9Q6CBJ2A9EDMFGY89AZZ4W5",
  "status": "running",
  "progress": {
    "total": 500,
    "completed": 214,
    "failed": 3
  },
  "started_at": "2026-01-10T15:04:05Z",
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

## `GET /v1/batch-evals/{batch_eval_id}/result` (result)

### Success response (200)

```json
{
  "batch_eval_id": "be_01J9Q6CBJ2A9EDMFGY89AZZ4W5",
  "status": "completed",
  "summary": {
    "mean_scores": {
      "accuracy": 7.8,
      "clarity": 8.2,
      "overall": 8.0
    },
    "total_items": 500,
    "failed_items": 4
  },
  "results_url": "https://example-bucket/results/be_01J9Q6CBJ2A9EDMFGY89AZZ4W5.jsonl",
  "completed_at": "2026-01-10T15:09:42Z",
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

---

## 7) Observability endpoints

## `GET /v1/metrics`
Prometheus/OpenMetrics-compatible plaintext response.

### Success response (200, text/plain)
Example metric names:
- `http_requests_total`
- `http_request_duration_ms`
- `inference_tokens_prompt_total`
- `inference_tokens_completion_total`
- `rate_limit_rejections_total`

## `GET /v1/health`
Liveness/readiness probe.

### Success response (200)

```json
{
  "status": "ok",
  "model": "distilgpt2",
  "device": "cpu"
}
```

---

## 8) Notes on rollout

- Endpoints marked here define the **target contract** for `v1`, even if some routes are staged and may return `501 not_implemented` during rollout.
- Clients should rely on versioned paths and tolerant parsing (ignore unknown fields) to remain forward-compatible in `v1`.
