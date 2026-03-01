# Evaluation Rubric

This rubric defines how to score model outputs in a deterministic, auditable way for batch and single-run evaluations.

## 1) Metric definitions

All metrics are computed per sample, then aggregated over an evaluation run.

### 1.1 Accuracy
- **Goal**: Measure task correctness against the reference answer or expected behavior.
- **Scale**: `0-2` integer.
  - `2` = fully correct, complete, and aligned with the expected answer.
  - `1` = partially correct (core idea present, but missing key detail or includes minor error).
  - `0` = incorrect or non-responsive.
- **Per-sample field**: `accuracy_score`.
- **Aggregate fields**:
  - `accuracy_mean` = mean of `accuracy_score` values.
  - `accuracy_full_credit_rate` = fraction with `accuracy_score = 2`.

### 1.2 Hallucination / Faithfulness
- **Goal**: Measure whether claims are grounded in provided context, tools, or references.
- **Scale**: `0-2` integer.
  - `2` = fully faithful; no unsupported claims.
  - `1` = mostly faithful; minor unsupported inference.
  - `0` = materially unfaithful or fabricated critical facts.
- **Per-sample field**: `faithfulness_score`.
- **Aggregate fields**:
  - `faithfulness_mean` = mean of `faithfulness_score`.
  - `faithfulness_failure_rate` = fraction with `faithfulness_score = 0`.

> Note: This rubric treats hallucination as the inverse of faithfulness for scoring simplicity.

### 1.3 Latency
- **Goal**: Capture responsiveness at user-perceived and model-only boundaries.
- **Per-sample fields**:
  - `latency_e2e_ms`: request start -> fully received final answer.
  - `latency_model_ms`: provider/model generation time when available.
- **Aggregate fields**:
  - `latency_e2e_p50_ms`, `latency_e2e_p95_ms`
  - `latency_model_p50_ms`, `latency_model_p95_ms`
- **Handling failures/timeouts**:
  - Record timed-out requests with `timed_out = true` and measured elapsed wall-clock time.

### 1.4 Token efficiency
- **Goal**: Measure cost-efficiency for prompt/response usage.
- **Per-sample fields**:
  - `input_tokens`
  - `output_tokens`
  - `total_tokens = input_tokens + output_tokens`
  - `token_efficiency_ratio = output_tokens / max(input_tokens, 1)`
- **Aggregate fields**:
  - `total_input_tokens`, `total_output_tokens`, `total_tokens`
  - `token_efficiency_ratio_mean`
  - `tokens_per_correct_answer = total_tokens / max(count(accuracy_score = 2), 1)`

---

## 2) Evaluator prompt/template and parsing expectations

Use this exact template for LLM-as-judge runs.

### 2.1 Evaluator prompt template (exact)

```text
You are a strict evaluation judge. Score the candidate answer using the rubric below.

RUBRIC
- accuracy_score (0,1,2):
  2 = fully correct and complete.
  1 = partially correct with minor error or omission.
  0 = incorrect, missing, or non-responsive.
- faithfulness_score (0,1,2):
  2 = all material claims grounded in provided context/tools/references.
  1 = mostly grounded with minor unsupported inference.
  0 = materially ungrounded or fabricated.

INPUTS
- task: {{task}}
- reference_answer: {{reference_answer}}
- provided_context: {{provided_context}}
- candidate_answer: {{candidate_answer}}

INSTRUCTIONS
1) Evaluate only the candidate_answer.
2) Do not reward style over correctness.
3) If context is insufficient, lower faithfulness when unsupported claims are asserted as facts.
4) Return strict JSON only (no markdown, no prose outside JSON).

OUTPUT JSON SCHEMA
{
  "accuracy_score": 0|1|2,
  "faithfulness_score": 0|1|2,
  "rationale": "<= 80 words"
}
```

### 2.2 Parsing expectations
- Response **must** be valid JSON object with keys:
  - `accuracy_score` (int in `{0,1,2}`)
  - `faithfulness_score` (int in `{0,1,2}`)
  - `rationale` (non-empty string, max 80 words)
- Reject and mark evaluator output as parse error if:
  - JSON is malformed.
  - Any required key is missing.
  - Any score is outside allowed values.
- Parse-error handling:
  - Retry evaluator once with identical inputs.
  - If second attempt fails, store:
    - `accuracy_score = null`
    - `faithfulness_score = null`
    - `evaluator_error = "parse_error"`

---

## 3) Pass/fail thresholds and weighting

### 3.1 Per-sample pass/fail
A sample **passes** only if all of the following are true:
- `accuracy_score >= 1`
- `faithfulness_score >= 1`
- `latency_e2e_ms <= 8000`
- `total_tokens <= 6000`

Otherwise, the sample **fails**.

### 3.2 Normalized component scores for aggregate scoring
For each sample, compute normalized components in `[0, 1]`:
- `accuracy_norm = accuracy_score / 2`
- `faithfulness_norm = faithfulness_score / 2`
- `latency_norm = min(1, 3000 / max(latency_e2e_ms, 1))`
- `token_efficiency_norm = min(1, 2000 / max(total_tokens, 1))`

### 3.3 Weighted aggregate score
Compute per-sample weighted score:

`sample_score = 0.45*accuracy_norm + 0.30*faithfulness_norm + 0.15*latency_norm + 0.10*token_efficiency_norm`

Run-level aggregate:

`aggregate_score = mean(sample_score)`

### 3.4 Run-level gates
A run is **release-ready** only if all conditions hold:
- `aggregate_score >= 0.80`
- `pass_rate >= 0.85`
- `faithfulness_failure_rate <= 0.05`
- `latency_e2e_p95_ms <= 10000`

---

## 4) Reproducibility controls

### 4.1 Generation defaults
Unless intentionally overridden for an experiment, use:
- `temperature = 0`
- `top_p = 1`
- `max_tokens = 1024`
- `seed = 42` (when provider supports seeded sampling)

### 4.2 Model version pinning
- Always record exact model identifier (for example: `provider/model-name@version`).
- Do not use floating aliases (for example `latest`) in benchmark runs.
- If provider does not expose semantic versions, capture provider-reported snapshot/build ID in metadata.

### 4.3 Required run metadata
Persist, at minimum:
- `run_id`
- `timestamp_utc`
- `dataset_id`
- `dataset_version_or_hash`
- `model_id`
- `model_version`
- `evaluator_model_id`
- `evaluator_model_version`
- `prompt_template_id`
- `prompt_template_version_or_hash`
- `evaluator_prompt_template_version_or_hash`
- `generation_params` (`temperature`, `top_p`, `max_tokens`, `seed`)
- `code_version` (git SHA)
- `environment` (region/runtime)

### 4.4 Repeatability protocol
- Run each benchmark at least `N=3` times for variance-sensitive tasks.
- Report mean and standard deviation for `aggregate_score` and `latency_e2e_p95_ms`.
- Store raw per-sample outputs to enable audit and rerun comparisons.
