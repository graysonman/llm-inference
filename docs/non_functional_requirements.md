# Non-Functional Requirements (NFRs)

This document defines production-readiness targets for the local LLM inference service and serves as a baseline for future staged hardening.

## 1) Performance

### Workload profile (test baseline)
- Endpoint: `POST /chat`
- Request profile:
  - Prompt length: 150-250 input tokens
  - `max_new_tokens`: 120
  - `temperature`: 0.2
- Model profile: default lightweight model (`distilgpt2`) unless explicitly overridden.
- Concurrency profile:
  - Sustained load test: **20 concurrent requests** for 10 minutes
  - Burst load test: **40 concurrent requests** for 60 seconds

### Performance targets
- Sustained p95 latency at 20 concurrent requests: **<= 2500 ms**
- Burst p95 latency at 40 concurrent requests: **<= 4000 ms**
- Throughput at 20 concurrent requests: **>= 8 requests/second**
- Throughput at 40 concurrent requests: **>= 10 requests/second**
- Timeout budget: <1% of requests may exceed **10 seconds**.

### Measurement requirements
- Measure on a single host with no competing CPU-intensive jobs.
- Capture latency histogram and requests/second for every load test run.
- Record model name, CPU/GPU class, and concurrency settings with every benchmark result.

## 2) Reliability

### Service objective assumptions (local deployment)
- Deployment target: single-node local container or VM.
- Planned downtime (upgrades/restarts) is excluded if announced in advance.
- Unplanned downtime includes process crashes, hang states, and dependency failures.

### Reliability targets
- Availability SLO (monthly): **99.0%** for local deployments.
- Error budget (monthly): **<= 1.0%** failed requests (5xx, timeout, or invalid model response).
- Crash recovery target:
  - Automatic restart in **<= 60 seconds** after process failure.
  - Health endpoint available in **<= 90 seconds** after restart.

### Failure-handling minimums
- Health checks must report degraded status when model load fails.
- On startup failures, service must return actionable error details in logs.
- Repeated failures (>=3 in 5 minutes) must trigger an operator-visible alert.

## 3) Security

### API key enforcement
- All non-health endpoints must require an API key in `Authorization: Bearer <key>`.
- Missing or invalid API keys must return **401 Unauthorized**.
- API keys must never be logged in plaintext.

### Key management and rotation policy
- Keys must be stored outside source code (environment variable or secret manager).
- Rotation cadence: **every 90 days** minimum.
- Emergency rotation must be executable in **<= 15 minutes**.
- At least one overlap window (old + new key valid) of **24 hours** is required to avoid abrupt client outage.

### Audit logging minimums
- Log security-relevant events:
  - Authentication success/failure
  - Key creation, revocation, and rotation actions
  - Configuration changes affecting auth or logging
- Audit records must include timestamp (UTC), actor/service identity, action, and outcome.
- Audit logs retention: **>= 90 days** locally, **>= 1 year** if exported to centralized storage.

## 4) Observability

### Required logs
- Per request:
  - Request ID
  - Endpoint
  - Status code
  - Latency (ms)
  - Prompt token count and completion token count
- Error logs must include stack trace and request ID.
- Security logs must be separated or clearly labeled for filtering.

### Required metrics
- Request rate (RPS)
- p50/p95/p99 latency
- 4xx and 5xx counts
- Timeout count
- Token usage (input/output)
- Model load duration and failure count
- Process-level CPU and memory usage

### Required traces
- Distributed trace (or equivalent span timing) for the full `/chat` request path.
- At minimum: request parsing, model inference, response serialization spans.
- Trace sampling:
  - 100% for errors
  - >=10% for successful requests

### Retention expectations
- Application logs: **>= 14 days** local retention.
- Metrics: high-resolution (<=15s) for **>= 14 days**, rolled-up aggregates for **>= 90 days**.
- Traces: **>= 7 days** retained.

## 5) Scalability

Thresholds that trigger migration from in-memory/local-only components to managed services:

- **Session/state storage**:
  - Move from in-memory to managed cache/store when any of:
    - >10,000 active sessions
    - >1 GB volatile state
    - multi-instance deployment required

- **Rate limiting / key metadata**:
  - Move from process memory to managed store when:
    - >100 API keys
    - >3 app instances
    - strict global rate limits are required across replicas

- **Logging pipeline**:
  - Move from local file logs to centralized logging when:
    - >5 GB logs/day
    - on-call requires cross-node querying
    - retention needs exceed local disk limits

- **Metrics and tracing backend**:
  - Move to managed observability backend when:
    - >3 services emit telemetry
    - >200k metric samples/minute
    - trace volume exceeds local collector capacity

- **Inference serving architecture**:
  - Move from single-process inference to dedicated model serving workers when:
    - sustained CPU >75% for 30 minutes
    - p95 latency breaches target for 3 consecutive load tests
    - throughput demand exceeds 2x current tested capacity

## Review cadence
- Reassess NFR targets at least once per quarter or whenever model/runtime architecture changes significantly.
