# Acceptance Scenarios

These are executable curl journeys for the currently implemented `v1` API.

## Conventions

```bash
export BASE_URL="http://localhost:8000"
export API_KEY="dev-local-key"
```

## 1) Upload dataset -> run batch eval -> inspect artifacts

```bash
# Upload an eval dataset file (.jsonl/.csv/.txt)
DATASET_ID=$(curl -s -X POST "$BASE_URL/v1/datasets/upload" \
  -H "x-api-key: $API_KEY" \
  -F "name=eval-fixture" \
  -F "type=eval_set" \
  -F "file=@./fixtures/eval_dataset.jsonl" | jq -r '.dataset_id')

# Start batch eval
RUN_ID=$(curl -s -X POST "$BASE_URL/v1/batch-evals" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"$DATASET_ID\",\"criteria\":[\"accuracy\",\"overall\"]}" | jq -r '.run_id')

# Read status/result/distribution/failures
curl -s "$BASE_URL/v1/batch-evals/$RUN_ID" -H "x-api-key: $API_KEY" | jq .
curl -s "$BASE_URL/v1/batch-evals/$RUN_ID/result" -H "x-api-key: $API_KEY" | jq .
curl -s "$BASE_URL/v1/batch-evals/$RUN_ID/distribution?criterion=overall" -H "x-api-key: $API_KEY" | jq .
curl -s "$BASE_URL/v1/batch-evals/$RUN_ID/failures?limit=20" -H "x-api-key: $API_KEY" | jq .
```

## 2) Upload corpus -> run RAG query -> inspect index status

```bash
RAG_DATASET_ID=$(curl -s -X POST "$BASE_URL/v1/datasets/upload" \
  -H "x-api-key: $API_KEY" \
  -F "name=rag-corpus" \
  -F "type=rag_corpus" \
  -F "file=@./fixtures/rag_corpus.jsonl" | jq -r '.dataset_id')

curl -s -X POST "$BASE_URL/v1/rag/query" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"$RAG_DATASET_ID\",\"query\":\"What does the policy say?\",\"top_k\":5}" | jq .

curl -s "$BASE_URL/v1/rag/indexes/$RAG_DATASET_ID" -H "x-api-key: $API_KEY" | jq .
```

## 3) Auth + rate limit behavior

```bash
# Missing key -> 401
curl -i -s -X POST "$BASE_URL/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}' | sed -n '1,20p'

# Valid key burst -> eventually 429
for i in $(seq 1 140); do
  code=$(curl -s -o /tmp/rate_$i.json -w "%{http_code}" -X POST "$BASE_URL/v1/chat" \
    -H "x-api-key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"rate-limit-test","max_new_tokens":16}')
  echo "$i -> $code"
done
```

## 4) Cache hit visibility + metrics dashboard

```bash
PROMPT="Summarize gradient descent in 2 bullets."

curl -s -X POST "$BASE_URL/v1/chat" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"max_new_tokens\":80}" | jq .

curl -s -X POST "$BASE_URL/v1/chat" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"max_new_tokens\":80}" | jq .

curl -s "$BASE_URL/metrics/dashboard?window=15m" -H "x-api-key: $API_KEY" | jq .
curl -s "$BASE_URL/v1/metrics?format=prometheus" -H "x-api-key: $API_KEY"
```
