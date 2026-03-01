# Dataset Contract (v1)

This document defines the canonical dataset shape and lifecycle behavior for batch evaluation inputs and outputs.

## 1) Accepted dataset JSON schema

### 1.1 Top-level dataset document

Datasets must be UTF-8 encoded JSON documents with a single top-level object.

```json
{
  "dataset_id": "qa_eval_set_2026_01",
  "dataset_version": "2026-01-15",
  "schema_version": "1.0",
  "created_at": "2026-01-15T10:05:12Z",
  "metadata": {
    "owner": "eval-team",
    "task_type": "qa",
    "source": "internal_benchmark"
  },
  "records": [
    {
      "record_id": "q_0001",
      "input": {
        "prompt": "Explain overfitting in two sentences."
      },
      "reference": {
        "answer": "Overfitting is when..."
      },
      "tags": ["ml", "definitions"],
      "expected": {
        "max_latency_ms": 2000,
        "required_criteria": ["accuracy", "clarity"]
      }
    }
  ]
}
```

### 1.2 Field requirements

#### Required top-level fields
- `dataset_id` (string)
  - Length: `1..128` characters.
  - Allowed chars: `A-Z`, `a-z`, `0-9`, `_`, `-`, `.`.
- `dataset_version` (string)
  - Length: `1..64` characters (semantic version, date, or release label).
- `schema_version` (string)
  - Must be exactly `"1.0"` for this contract version.
- `records` (array)
  - Minimum items: `1`.
  - Maximum items per uploaded dataset: `50,000`.

#### Optional top-level fields
- `created_at` (string, ISO-8601 UTC timestamp).
- `metadata` (object)
  - Maximum serialized size: `16 KB`.
  - Max depth: 5.

#### Required per-record fields
- `record_id` (string)
  - Length: `1..128` characters.
  - Must be unique within `records`.
- `input` (object)
  - Must include `prompt` (string).
- `input.prompt` (string)
  - Length: `1..200,000` characters.

#### Optional per-record fields
- `reference` (object)
  - May include `answer` (string, max `200,000` chars) and task-specific reference data.
- `tags` (array of strings)
  - Max items: `32`.
  - Each tag length: `1..64`.
- `expected` (object)
  - `max_latency_ms` (integer, `1..120000`).
  - `required_criteria` (array of enum): `accuracy | clarity | reasoning | factuality | overall`.
- `metadata` (object)
  - Max serialized size: `8 KB`.
  - Max depth: 5.

### 1.3 Size limits and encoding assumptions

- Transport content type: `application/json`.
- Character encoding: UTF-8 only.
- UTF-8 BOM is accepted and ignored.
- Newlines may be `\n` or `\r\n`.
- Maximum uploaded payload size: `100 MB`.
- Maximum serialized size per single record object: `256 KB`.
- Clients must normalize string values to Unicode NFC before upload.
- Null bytes (`\u0000`) and unpaired surrogate code points are rejected.

---

## 2) Validation rules and malformed-record error format

Validation occurs in two phases:
1. **Dataset-level validation** (top-level shape, limits, parseability).
2. **Record-level validation** (applied independently to each `records[i]`).

### 2.1 Validation rules

#### Dataset-level hard failures (request rejected)
- Invalid JSON, non-object top level, or non-UTF-8 payload.
- Missing required top-level fields.
- `schema_version` not supported.
- `records` exceeds max count.
- Payload exceeds max size.

These return HTTP `400` (`invalid_request`) or `413` (`payload_too_large`) and no run is created.

#### Record-level failures (run can continue)
Each record is validated for:
- required fields present and typed correctly;
- length/range checks;
- enum validity;
- duplicate `record_id` detection;
- forbidden control characters in text;
- object depth and serialized size limits.

Invalid records are marked failed with status `invalid_record`, and valid records continue through evaluation.

### 2.2 Exact malformed-record error response format

When one or more records are malformed but the dataset is otherwise accepted, the API returns HTTP `202` with a run object and explicit validation errors:

```json
{
  "run_id": "run_01J8Q4ER9Z7Y5V5K6F4A1P2M3N",
  "status": "accepted_with_record_errors",
  "summary": {
    "total_records": 3,
    "accepted_records": 2,
    "rejected_records": 1
  },
  "record_errors": [
    {
      "index": 1,
      "record_id": "q_0002",
      "code": "invalid_field_type",
      "message": "input.prompt must be a string",
      "path": "records[1].input.prompt",
      "severity": "error"
    }
  ],
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

`record_errors[]` fields are fixed as:
- `index` (integer, zero-based record index)
- `record_id` (string or `null` if unavailable)
- `code` (enum)
- `message` (human-readable string)
- `path` (JSONPath-like field location)
- `severity` (currently always `error`)

Allowed `code` values:
- `missing_required_field`
- `invalid_field_type`
- `value_out_of_range`
- `string_too_long`
- `invalid_enum_value`
- `duplicate_record_id`
- `record_too_large`
- `invalid_encoding`
- `unsupported_field`

If all records are malformed, the response is HTTP `400` with standard error envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "All records failed validation",
    "details": {
      "rejected_records": 100,
      "accepted_records": 0
    }
  },
  "request_id": "8ce7e5f9-7a47-49fd-bb9f-7d0ddc49e1e0"
}
```

---

## 3) Batch evaluation run model

### 3.1 Run states

A run transitions through the following states:

1. `queued` - run accepted and waiting for workers.
2. `validating` - dataset and records are being validated.
3. `running` - valid records are being evaluated.
4. `retrying` - transient failures are being retried.
5. `finalizing` - aggregation and artifact persistence in progress.
6. Terminal states:
   - `completed` - all valid records processed successfully.
   - `completed_with_failures` - at least one record failed permanently.
   - `failed` - unrecoverable run-level failure (for example, storage outage).
   - `cancelled` - explicit user/system cancellation.

### 3.2 Retry policy

Retries apply only to transient errors (`timeout`, `rate_limited`, `service_unavailable`, `internal_error`).

- Max attempts per record: `3` total (`1` initial + `2` retries).
- Backoff schedule: exponential with jitter (`2s`, `6s`, `14s`, ±20%).
- Retry is skipped for validation and other permanent errors.
- Each attempt is persisted with timestamp, error code, and latency.

### 3.3 Partial-failure behavior

- Record evaluation is isolated; one record failure does not stop other records.
- Final run status logic:
  - `completed` if `failed_records = 0` and `cancelled_records = 0`.
  - `completed_with_failures` if any record is `invalid_record`, `timeout`, `evaluation_error`, or `cancelled`.
  - `failed` only for run-level blocking issues before meaningful processing.
- Aggregate metrics are computed from successfully evaluated records and include denominators:
  - `total_records`
  - `valid_records`
  - `evaluated_records`
  - `failed_records`
  - `skipped_records`

---

## 4) Stored artifacts per run

Each run must persist immutable artifacts for auditability and reproducibility.

### 4.1 Required artifact set

- `run_manifest.json`
  - run identifiers, schema versions, status, and state timestamps.
- `input_dataset.json`
  - normalized accepted input payload (or content-addressed reference).
- `record_validation.jsonl`
  - per-record validation result (`accepted` or error details).
- `predictions.jsonl`
  - per-record model outputs and evaluator outputs.
- `attempt_logs.jsonl`
  - per-attempt request/response metadata, retries, and error codes.
- `metrics_summary.json`
  - aggregate metrics and confidence intervals.
- `metrics_by_slice.json`
  - metrics grouped by configured slices (for example `tags`, task type).
- `failures.jsonl`
  - normalized permanent failures with taxonomy labels.

### 4.2 Required fields across artifacts

#### Inputs
- `dataset_id`, `dataset_version`, `schema_version`
- record payload hash (`record_sha256`)

#### Outputs
- `record_id`
- `model_response`
- `evaluator_scores` / rubric outputs
- `output_tokens`, `total_tokens`, latency measures

#### Metrics
- pass/fail counts and rates
- quality score distributions
- latency p50/p95
- cost and token aggregates

#### Timestamps
- `created_at`
- `started_at`
- `completed_at`
- per-record `first_attempt_at`, `last_attempt_at`

#### Model parameters
- `model` (exact identifier)
- generation settings (`temperature`, `top_p`, `max_new_tokens`, `seed` when set)
- evaluator version and prompt/template version/hash

### 4.3 Retention and immutability

- Artifacts are write-once per `run_id`.
- Corrections create a new run/version, not in-place mutation.
- Minimum retention: `30` days for raw artifacts, `180` days for aggregate summaries (policy may be stricter by environment).
- Sensitive fields (PII/secrets) must be redacted before persistence.
