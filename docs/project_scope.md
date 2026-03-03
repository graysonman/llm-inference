# Project Scope

This document defines what is required to complete the current phase of the project, what is aspirational, and what is intentionally deferred.

## 1) Must-have (required for completion)

The following capabilities are required for this phase to be considered complete:

- Chat
- Evaluate
- Embeddings
- RAG
- Dataset upload
- Batch eval
- Persistence
- Auth
- Redis cache
- Metrics dashboard

## 2) Stretch goals

The following are valuable but optional for this phase:

- FAISS swap
- React migration
- Advanced agent tools
- Tracing backend integrations

## 3) Out-of-scope for current phase

The following are explicitly out of scope right now:

- Multimodal
- Fine-tuning pipeline
- Distributed serving

## 4) Done-when checklist

Use this checklist to determine phase completion status.

### Must-have (completion gate)

_Status updated: 2026-03-02_

- [x] Chat is implemented and usable end-to-end.
- [x] Evaluate functionality is implemented and runnable.
- [x] Embeddings pipeline is implemented and callable.
- [x] RAG workflow is implemented with retrieval + generation.
- [x] Dataset upload is implemented with basic validation.
- [x] Batch eval is implemented for multi-item evaluation runs.
- [x] Persistence is implemented for core project state/data. (JSON snapshot and optional SQLite backend)
- [x] Auth is implemented for protected access paths.
- [x] Redis cache is integrated and used in relevant hot paths. (with in-memory fallback)
- [x] Metrics dashboard is available with core operational metrics.

### Stretch goals (nice-to-have)

- [ ] FAISS swap path is available and documented.
- [ ] React migration milestone is completed for targeted UI surfaces.
- [ ] Advanced agent tools are integrated for selected workflows.
- [ ] Tracing backend integration is connected and verified.

### Out-of-scope guardrails (must remain deferred this phase)

- [x] Multimodal support is not committed as part of this phase plan.
- [x] Fine-tuning pipeline work is deferred to a future phase.
- [x] Distributed serving work is deferred to a future phase.
