# ADR 0002: Vector Store Path (In-Memory Baseline, FAISS/Redis Promotion)

- **Status:** Accepted
- **Date:** 2026-02-28

## Context

The system needs semantic retrieval with vector similarity search. Early-stage workloads are expected to be modest, but we need a path to low-latency retrieval at larger corpus sizes and query rates.

## Decision

Start with an **in-memory vector index baseline**, and promote to **FAISS or Redis vector indexing** based on defined scale and operational criteria.

### Baseline (in-memory)

- Store embeddings in process memory for initial deployments.
- Use simple nearest-neighbor search suitable for low/medium corpus sizes.
- Rebuild index on startup from persisted source data.

### Promotion targets

- **FAISS** for single-node, high-performance ANN search where local disk/memory optimization is preferred.
- **Redis vector search** when a managed/distributed service model and operational simplicity across replicas is preferred.

## Promotion Criteria (In-Memory → FAISS/Redis)

Promote when one or more criteria are met:

1. Corpus size or embedding dimensionality causes unacceptable memory pressure in the app process.
2. P95/P99 retrieval latency breaches retrieval SLO under target traffic.
3. Startup reindex time exceeds acceptable deploy/restart windows.
4. Need for shared index access across multiple app instances.

Selection guidance:

- Choose **FAISS** when maximizing retrieval performance on dedicated hosts is the primary objective.
- Choose **Redis** when operational consistency, remote access, and shared indexing across services are primary objectives.

## Consequences

### Positive

- Minimal dependencies and fast iteration initially.
- Objective, metric-driven promotion path.
- Avoids over-engineering before scale requirements are real.

### Negative / Trade-offs

- In-memory index limits horizontal scalability.
- Promotion requires index migration/rebuild and operational runbooks.
- FAISS and Redis each introduce distinct operational complexity.

## Implementation Notes

- Define retrieval SLOs (e.g., P95 latency) early.
- Instrument index size, query latency, and reindex duration.
- Keep embedding generation and storage interfaces backend-agnostic.
