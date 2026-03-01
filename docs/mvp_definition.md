# MVP Definition

This document defines the minimum release scope for the first production-ready milestone. Anything not listed as **Required (MVP)** is **Post-MVP** by default.

## 1) Required (MVP) vs Post-MVP Feature Scope

### Backend (Required for MVP — must ship)

- **Authentication and authorization**
  - Basic login/session or token flow for protected API access.
  - Role/scope checks for user-only project data access.
- **Core chat API**
  - Create chat session, send prompt, receive model response.
  - Persist chat history per authenticated user.
- **Evaluation API (single + batch)**
  - Submit one evaluation job and retrieve result.
  - Submit batch evaluation and track job status/results.
- **Embeddings + RAG API path**
  - Create embeddings for uploaded documents.
  - Retrieval endpoint powering generation with cited context chunks.
- **Dataset/document upload**
  - Upload endpoint with file type/size validation.
  - Metadata persistence and retrieval for uploaded assets.
- **Persistence + cache foundation**
  - Durable storage for chats, datasets, and evaluation outputs.
  - Redis cache on agreed hot paths.
- **Operational visibility**
  - Health/readiness endpoints.
  - Core metrics exposed for dashboarding.

### UI (Required for MVP — must ship)

- **Auth flow UI**
  - Sign-in/sign-out and protected route behavior.
- **Chat UI**
  - Prompt input, streamed/non-streamed response display, conversation history.
- **Evaluation UI**
  - Single-run evaluation trigger + result display.
  - Batch run creation + status/result table.
- **Dataset upload UI**
  - Upload form, validation feedback, and dataset listing.
- **RAG interaction UI**
  - Query interface that uses retrieval-backed responses.
  - Basic source/citation presentation in results.
- **MVP observability surface**
  - Simple internal metrics/status page or linked dashboard access point.

### Post-MVP (everything else)

- React migration or framework rewrite.
- FAISS storage-engine swap and alternative vector index experiments.
- Advanced agent tooling/orchestration features.
- Third-party tracing backend integrations beyond baseline metrics.
- Multimodal inference.
- Fine-tuning/training pipeline.
- Distributed serving and multi-region deployment.
- Any UX polish not required for completing MVP user journeys above.

## 2) Hard De-Scope Rules

The following rules are mandatory and override preference-based roadmap requests:

1. **No React migration before all required API endpoints are stable** (chat, eval, batch eval, upload, embeddings/RAG) and passing acceptance criteria.
2. **No new model/provider integrations** before MVP reliability gates are met for the default provider path.
3. **No multimodal scope** (image/audio/video) in MVP planning, implementation, or acceptance sign-off.
4. **No fine-tuning features** until MVP launch criteria are met and post-launch capacity is approved.
5. **No distributed serving work** before single-node production readiness is validated.
6. **No major UI redesign** that delays required MVP flows; UX changes must be strictly incremental for MVP.
7. **No “one-off exceptions”** to add features not listed under Required (MVP) without an owner decision log entry.

## 3) Release Blockers vs Nice-to-Have

### Release blockers (must be resolved before release)

- Any Required (MVP) backend or UI feature is incomplete.
- Critical auth/data isolation defect.
- Chat, evaluation, upload, or RAG endpoints fail agreed acceptance scenarios.
- Data persistence failures for core entities (chat history, uploads, eval results).
- P0/P1 reliability issues affecting core user journeys.
- Missing health checks/metrics needed for safe operation.

### Nice-to-have (not a release gate)

- UI polish improvements beyond functional clarity.
- Additional visualizations for metrics dashboards.
- Performance optimization beyond agreed MVP baseline.
- Alternative vector backends and tracing integrations.
- Framework migration groundwork.

## 4) Owner Decision Log (single source for scope disputes)

Use this section as the only authoritative record for scope disputes and exceptions.

| Date (YYYY-MM-DD) | Decision Owner | Scope Question / Dispute | Decision | Rationale | Impact (Timeline/Quality) | Follow-up Action |
|---|---|---|---|---|---|---|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Decision log rules:**

- One decision owner per entry (no shared ownership).
- If a request conflicts with this MVP definition, it is blocked until recorded here.
- Latest dated decision wins unless explicitly superseded by a newer entry.
