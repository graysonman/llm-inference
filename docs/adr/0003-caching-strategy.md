# ADR 0003: Caching Strategy (Scope, TTL Policy, Invalidation Rules)

- **Status:** Accepted
- **Date:** 2026-02-28

## Context

We need to reduce end-to-end latency and repeated compute/storage access without introducing correctness issues from stale data.

## Decision

Adopt a **tiered caching strategy** with explicit TTL defaults and deterministic invalidation rules.

## What Gets Cached

1. **Retrieval results** (query → candidate IDs + scores) for repeated semantic queries.
2. **Embedding outputs** (text hash/model version → vector) to avoid recomputation.
3. **Read-most metadata** (document metadata/config snapshots) with short-lived cache.
4. **Expensive downstream responses** only when idempotent and safe to reuse.

## TTL Policy

- Default TTL: **5 minutes** for retrieval/result caches.
- Embedding cache TTL: **24 hours** (or longer) keyed by model/version/content hash.
- Metadata/config cache TTL: **1–5 minutes** depending on update frequency.
- Negative-cache entries (misses/errors): **30–60 seconds** to prevent hot-loop retries.

All TTLs are configurable via environment/application settings.

## Invalidation Rules

1. **Content mutation invalidates dependent caches**:
   - On create/update/delete of documents, invalidate retrieval and metadata keys for affected scopes.
2. **Model/version change invalidates embedding caches**:
   - Versioned keys ensure automatic rollover; optional bulk purge on major upgrades.
3. **Configuration changes invalidate relevant computed responses**:
   - Include config revision in cache key where feasible.
4. **Failure-aware invalidation**:
   - Do not cache transient upstream failures beyond short negative-cache TTL.

## Consequences

### Positive

- Better latency and reduced repeated compute.
- Clear data freshness behavior with bounded staleness.
- Operationally tunable without code changes.

### Negative / Trade-offs

- Requires disciplined key design and observability.
- Overly aggressive invalidation can reduce hit rates.
- Incorrect invalidation can serve stale responses.

## Implementation Notes

- Track cache hit/miss rates per cache domain.
- Add cardinality controls and max-size limits.
- Use namespaced keys: `{domain}:{version}:{scope}:{hash}`.
