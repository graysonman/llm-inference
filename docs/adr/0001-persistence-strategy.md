# ADR 0001: Persistence Strategy (SQLite-First, Postgres Migration Path)

- **Status:** Accepted
- **Date:** 2026-02-28

## Context

We need a persistence layer that is simple to run in local/dev environments while still supporting growth into multi-instance production deployments.

Current constraints:

- We want low operational overhead for early iterations.
- We need SQL semantics and transactional consistency for metadata and operational state.
- We anticipate a future need for stronger concurrency and managed backup/replication options.

## Decision

Adopt a **SQLite-first strategy** for initial implementation, with explicit compatibility boundaries to enable migration to **Postgres** when scale or operational requirements justify it.

### SQLite baseline

- SQLite is the default database engine for local/dev and initial production pilots.
- Schema is managed through versioned migrations (forward and rollback scripts).
- Use a SQL subset that remains portable to Postgres (avoid engine-specific SQL unless isolated).

### Postgres migration path

- Keep SQL access behind a repository/DAO abstraction to reduce migration blast radius.
- Use UUID/text identifiers and explicit indexes that map cleanly to both engines.
- Add migration readiness checks (e.g., feature flags and smoke tests) before promoting Postgres.
- Provide a one-time data export/import path from SQLite to Postgres.

## Promotion Criteria (SQLite → Postgres)

Promote when any of the following are consistently observed:

1. Sustained write contention or lock wait impact from concurrent writers.
2. Need for horizontal scaling across multiple application replicas.
3. Operational requirements for managed HA, point-in-time recovery, or read replicas.
4. Data size growth where SQLite backup/maintenance windows become unacceptable.

## Consequences

### Positive

- Fast developer onboarding and minimal infra dependencies early on.
- Lower operational cost and complexity in initial phases.
- Clear path to scale without rewriting domain logic.

### Negative / Trade-offs

- Must enforce SQL portability discipline from day one.
- Migration introduces operational work (cutover, validation, monitoring).
- Some Postgres-specific optimizations are deferred until promotion.

## Implementation Notes

- Use migration tooling from the start, even with SQLite.
- Include data access integration tests that can run against both SQLite and Postgres.
- Capture DB metrics (lock time, query latency, write throughput) to support objective promotion decisions.
