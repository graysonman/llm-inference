# Evaluation Specification

This document defines the required metrics, aggregation methods, and run metadata for evaluating model inference behavior across latency, quality, cost, and reliability dimensions.

## 1) Latency metrics

Record latency in milliseconds (`ms`) for every evaluated sample.

### 1.1 End-to-end latency
- **Definition**: Time from request dispatch to fully received final response (including network overhead, retries, middleware, and post-processing).
- **Required percentiles**:
  - `e2e_latency_p50_ms`
  - `e2e_latency_p95_ms`
- **Collection guidance**:
  - Measure wall-clock duration at the client boundary.
  - Use monotonic clocks.
  - Include failed calls in a separate failure-latency distribution.

### 1.2 Model-only latency
- **Definition**: Time attributable to model generation only (exclude client-side queueing, serialization, and downstream post-processing where separable).
- **Required percentiles**:
  - `model_latency_p50_ms`
  - `model_latency_p95_ms`
- **Collection guidance**:
  - Prefer provider-reported generation latency if available.
  - If unavailable, approximate with server-side timings and document approximation logic.

---

## 2) Quality metrics

Every sample must receive evaluator labels on correctness, faithfulness, and hallucination.

### 2.1 Correctness rubric
Score the final answer on a 0-2 scale:
- **2 (Correct)**: Fully correct and complete for the task requirements.
- **1 (Partially correct)**: Contains meaningful correct content but is incomplete, ambiguous, or has minor errors.
- **0 (Incorrect)**: Fails to answer the task or contains major errors.

### 2.2 Faithfulness rubric
Assesses alignment with provided context/tool outputs/source evidence.
- **2 (Faithful)**: Claims are supported by provided evidence/context.
- **1 (Weakly faithful)**: Mostly grounded, but includes unsupported inference or overstated certainty.
- **0 (Unfaithful)**: Material claims contradict or are not grounded in available evidence.

### 2.3 Hallucination rubric
Assesses fabrication of facts, citations, or entities.
- **0 (None)**: No detectable fabrication.
- **1 (Minor)**: Non-critical fabricated detail that does not change core conclusion.
- **2 (Major)**: Critical fabrication that changes interpretation or outcome.

### 2.4 Evaluator output schema (per sample)
- `correctness_score` in `{0,1,2}`
- `faithfulness_score` in `{0,1,2}`
- `hallucination_score` in `{0,1,2}`
- `evaluator_notes` (free text rationale)

---

## 3) Cost and efficiency metrics

Track token utilization and efficiency per sample and aggregate at dataset level.

### 3.1 Required token fields
- `input_tokens`
- `output_tokens`
- `total_tokens` (= `input_tokens + output_tokens`)

### 3.2 Output-input ratio
- **Definition**: `output_input_ratio = output_tokens / max(input_tokens, 1)`
- **Purpose**: Detect unusually verbose responses and prompt-response imbalance.

### 3.3 Cache savings
Capture token/cost savings from prompt caching mechanisms.
- Required fields:
  - `cache_read_input_tokens` (reused input tokens)
  - `cache_write_input_tokens` (newly cached input tokens)
  - `estimated_cache_savings_tokens`
- If direct token savings are unavailable, store `estimated_cache_savings_usd` with estimation method documented.

---

## 4) Dataset-level aggregates

Compute aggregates over the full evaluation set and any key slices (model, task type, prompt template version).

### 4.1 Pass rate
- **Definition**: Fraction of samples meeting a defined success condition.
- **Default success condition**: `correctness_score >= 2` and `hallucination_score = 0`.
- Report:
  - `pass_rate`
  - `pass_count`
  - `total_count`

### 4.2 Failure taxonomy
Each failed sample must be assigned one primary failure label (optionally multiple secondary labels).

Suggested primary taxonomy:
- `incorrect_answer`
- `missing_required_content`
- `unfaithful_to_context`
- `hallucinated_fact`
- `format_or_schema_violation`
- `tool_or_retrieval_misuse`
- `timeout_or_latency_exceeded`
- `other`

Report counts and percentages per label.

### 4.3 Confidence bands
Provide uncertainty estimates for key rates (at minimum pass rate).
- Required: 95% confidence interval (CI)
- Preferred methods:
  - Wilson interval for binomial rates
  - Bootstrap CI for non-binary aggregates (e.g., mean latency, mean scores)
- Required fields for pass rate:
  - `pass_rate_ci95_lower`
  - `pass_rate_ci95_upper`

---

## 5) Required metadata saved per run

Every evaluation run must persist the following metadata for reproducibility and auditing:

- `run_id` (unique identifier)
- `timestamp_utc` (ISO-8601)
- `model` (exact provider/model identifier)
- `params` (generation parameters such as temperature, top_p, max_tokens, seed, etc.)
- `prompt_template` (template id/version and resolved template text or hash)
- `api_key_id` (non-secret identifier only; never store raw API keys)

Recommended additional metadata:
- `code_version` (git commit SHA)
- `dataset_id` and `dataset_version`
- `evaluator_version` (manual rubric version or automated evaluator model/version)
- `environment` (region, deployment, runtime)

---

## Storage and reporting requirements

- Persist both **per-sample records** and **dataset-level aggregate summaries**.
- Use stable schema names and version them (e.g., `evaluation_schema_version`).
- Ensure personally identifying information and raw secrets are excluded from logs and artifacts.
