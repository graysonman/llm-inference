# Acceptance Scenarios: Executable User Journeys

This document provides executable, end-to-end acceptance scenarios for the platform roadmap.
Each scenario is written as a reproducible shell journey using `curl` and `jq`.

## Conventions

- API base URL: `http://localhost:8000`
- Auth token variable: `$TOKEN`
- JSON parsing: `jq`

```bash
export BASE_URL="http://localhost:8000"
export TOKEN="replace-with-valid-bearer-token"
```

---

## 1) Upload dataset → run batch eval → view score distribution and failures

### Goal
Validate that an evaluation dataset can be uploaded, evaluated in batch, and inspected for distribution + failed rows.

### Journey

```bash
# 1) Upload evaluation dataset
EVAL_DATASET_ID=$(curl -s -X POST "$BASE_URL/evals/datasets" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./fixtures/eval_dataset.jsonl" | jq -r '.dataset_id')

echo "Dataset ID: $EVAL_DATASET_ID"

# 2) Start batch evaluation run
EVAL_RUN_ID=$(curl -s -X POST "$BASE_URL/evals/runs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"$EVAL_DATASET_ID\",\"metric\":\"answer_correctness\"}" | jq -r '.run_id')

echo "Run ID: $EVAL_RUN_ID"

# 3) Wait for completion
until [ "$(curl -s "$BASE_URL/evals/runs/$EVAL_RUN_ID" -H "Authorization: Bearer $TOKEN" | jq -r '.status')" = "completed" ]; do
  sleep 2
  echo "Waiting for run completion..."
done

# 4) Inspect score distribution
curl -s "$BASE_URL/evals/runs/$EVAL_RUN_ID/distribution" \
  -H "Authorization: Bearer $TOKEN" | jq .

# 5) Inspect failed examples
curl -s "$BASE_URL/evals/runs/$EVAL_RUN_ID/failures?limit=20" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

### Acceptance checks
- Dataset upload returns a non-empty `dataset_id`.
- Evaluation run reaches `completed`.
- Distribution endpoint returns histogram/buckets and summary stats.
- Failures endpoint returns only failed rows with reason and trace metadata.

---

## 2) Upload docs → run RAG query → inspect retrieved chunks and latency

### Goal
Verify document ingestion feeds retrieval, and query responses expose retrieval evidence and timing.

### Journey

```bash
# 1) Upload one or more source documents
CORPUS_ID=$(curl -s -X POST "$BASE_URL/rag/corpora" \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@./fixtures/handbook.md" \
  -F "files=@./fixtures/runbook.pdf" | jq -r '.corpus_id')

echo "Corpus ID: $CORPUS_ID"

# 2) Run a RAG query
curl -s -X POST "$BASE_URL/rag/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"corpus_id\":\"$CORPUS_ID\",\"question\":\"What is the incident escalation policy?\",\"top_k\":5}" \
  | tee /tmp/rag_response.json | jq .

# 3) Inspect retrieved chunks and latency
jq '.retrieval.chunks[] | {document_id, chunk_id, score, text}' /tmp/rag_response.json
jq '.metrics | {total_latency_ms, retrieval_latency_ms, generation_latency_ms}' /tmp/rag_response.json
```

### Acceptance checks
- Upload returns non-empty `corpus_id`.
- Query response includes answer text and retrieval metadata.
- At least one retrieved chunk is present with score and source reference.
- Metrics include total and stage-level latency.

---

## 3) Authenticated requests only → enforce rate limit behavior

### Goal
Ensure protected endpoints reject unauthenticated traffic and throttle excess authenticated requests.

### Journey

```bash
# 1) Unauthenticated request should fail with 401/403
curl -i -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}' | sed -n '1,12p'

# 2) Authenticated burst should eventually trigger 429
for i in $(seq 1 30); do
  code=$(curl -s -o /tmp/rate_$i.json -w "%{http_code}" -X POST "$BASE_URL/chat" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"rate-limit-test","max_new_tokens":16}')
  echo "$i -> $code"
done

# 3) Verify rate-limit headers / body when throttled
last_429=$(for i in $(seq 1 30); do
  code=$(curl -s -o /tmp/rate_$i.json -w "%{http_code}" -X POST "$BASE_URL/chat" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"rate-limit-test","max_new_tokens":16}')
  [ "$code" = "429" ] && echo "$i" && break
done)

echo "First 429 at request: $last_429"
[ -n "$last_429" ] && cat "/tmp/rate_${last_429}.json" | jq .
```

### Acceptance checks
- Protected endpoint denies missing token with 401/403.
- Authenticated burst yields at least one 429.
- 429 response includes retry metadata (`Retry-After`, limit window, or equivalent).

---

## 4) Repeated queries show cache-hit improvements in metrics dashboard

### Goal
Demonstrate that repeated identical requests become faster and increase cache-hit counters.

### Journey

```bash
PROMPT='Summarize the concept of gradient descent in 2 bullets.'

# 1) Warm-up request (expected cache miss)
curl -s -X POST "$BASE_URL/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"max_new_tokens\":80}" \
  | tee /tmp/cache_req_1.json | jq '.metrics'

# 2) Repeat same request several times (expected cache hits)
for i in 2 3 4 5; do
  curl -s -X POST "$BASE_URL/chat" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"$PROMPT\",\"max_new_tokens\":80}" \
    | tee "/tmp/cache_req_${i}.json" | jq '.metrics'
done

# 3) Inspect dashboard metrics for cache hit ratio + latency trend
curl -s "$BASE_URL/metrics/dashboard?window=5m" \
  -H "Authorization: Bearer $TOKEN" | jq '.cache, .latency'
```

### Acceptance checks
- First response marks cache miss (or no hit flag).
- Subsequent identical responses indicate cache hits.
- Dashboard reflects increased cache-hit ratio over the test window.
- Median/p95 latency improves after warm-up.

---

## 5) Concurrent request simulation demonstrates stable service and tracked queue depth

### Goal
Stress the service with concurrent load and verify stability + queue observability.

### Journey

```bash
# 1) Run included stress test at moderate concurrency
python scripts/stress_test.py \
  --url "$BASE_URL/chat" \
  --token "$TOKEN" \
  --concurrency 20 \
  --requests 200 \
  --prompt "Return one sentence on reliability engineering."

# 2) Pull service health + queue depth metrics during/after load
curl -s "$BASE_URL/health" -H "Authorization: Bearer $TOKEN" | jq .
curl -s "$BASE_URL/metrics/dashboard?window=15m" -H "Authorization: Bearer $TOKEN" \
  | jq '.service, .queue_depth, .errors, .latency'
```

### Acceptance checks
- Service remains responsive throughout load test.
- Error rate stays within SLO threshold (for example, <1%).
- Queue depth is captured in metrics and returns to baseline after load subsides.
- No crash/restart indicators in health or service metrics.

---

## Notes for CI / Automation

- These journeys are suitable for nightly acceptance pipelines once endpoints are enabled.
- Fixture files (`./fixtures/*`) should be versioned and deterministic.
- CI should fail fast on:
  - missing IDs (`dataset_id`, `run_id`, `corpus_id`)
  - non-`completed` eval status
  - absent retrieval chunks/latency metrics
  - no observed `429` during throttling test
  - degraded stability or missing queue-depth telemetry
