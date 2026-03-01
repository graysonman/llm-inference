# ADR 0004: Async/Concurrency Approach (Threadpool Boundaries, Queue Metrics, Failure Handling)

- **Status:** Accepted
- **Date:** 2026-02-28

## Context

Workloads mix network-bound calls (model APIs, storage) and CPU-heavy tasks (embedding prep, ranking logic). We need predictable concurrency, backpressure, and robust failure behavior.

## Decision

Use an **async-first request path** with explicit threadpool boundaries for CPU-bound work and bounded queues for backpressure.

## Concurrency Model

1. **Async runtime for I/O-bound operations**:
   - External API calls, database reads/writes, cache/network operations.
2. **Dedicated CPU threadpool for CPU-bound tasks**:
   - Embedding post-processing, ranking/scoring transforms, serialization-heavy operations.
3. **Bounded work queues** between stages:
   - Prevent unbounded memory growth and enforce load shedding when saturated.

## Threadpool Boundaries

- Do not run CPU-heavy jobs on async executor worker threads.
- Route CPU tasks through dedicated executors with max worker limits.
- Keep blocking DB/file operations off async worker threads unless driver/runtime guarantees non-blocking behavior.

## Queue Metrics & SLO Signals

Collect and alert on:

- Queue depth (current/max).
- Queue wait time (P50/P95/P99).
- Task execution time by stage.
- Rejection/drop rate due to backpressure.
- End-to-end request latency and timeout rate.

## Failure Handling

1. **Timeouts per stage** with bounded retries and exponential backoff.
2. **Circuit-breaking** for repeatedly failing upstream dependencies.
3. **Idempotent retry semantics** for safe operations only.
4. **Graceful degradation**:
   - Return partial/fallback results when non-critical enrichment stages fail.
5. **Dead-letter handling** for repeatedly failing background jobs.

## Consequences

### Positive

- Better tail-latency control under load.
- Reduced risk of executor starvation.
- Observable and tunable backpressure behavior.

### Negative / Trade-offs

- Additional operational complexity (queues, metrics, tuning).
- Requires careful capacity planning for threadpools and queue limits.
- More nuanced failure modes across async stage boundaries.

## Implementation Notes

- Start with conservative queue bounds and tune using production telemetry.
- Define per-stage timeout budgets that add up to end-to-end SLO.
- Document retry safety per operation to avoid duplicate side effects.
