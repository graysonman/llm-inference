# UI Architecture

This document defines the required product screens/tabs for the LLM inference UI and the core information architecture for each.

## 1) Playground

**Purpose:** Interactive inference and prompt experimentation.

### Core regions
- **Chat panel**
  - User/system message input
  - Conversation history with role labels
  - Streaming assistant response display
- **Model parameters panel**
  - Model selector
  - Temperature
  - Top-p
  - Max output tokens
  - Optional stop sequences
- **Performance summary panel**
  - End-to-end latency (ms)
  - Input/output/total tokens
  - Tokens per second (if available)

### Key interactions
- Send a prompt and receive streamed or non-streamed response.
- Adjust parameters and re-run quickly.
- Compare response quality against latency/token cost.

---

## 2) RAG Explorer

**Purpose:** Inspect retrieval quality and grounding behavior for retrieval-augmented generation.

### Core regions
- **Document upload and indexing**
  - Upload one or more source documents
  - Show indexing status and ingest errors
- **Retrieved chunks viewer**
  - Ranked list of retrieved chunks
  - Chunk source metadata (document name, page/section)
- **Similarity score panel**
  - Score per retrieved chunk
  - Ordering by rank and threshold visibility
- **Groundedness hints**
  - Heuristics/indicators linking response claims to retrieved chunks
  - Missing citation or weak grounding warnings

### Key interactions
- Upload documents, run a query, and inspect top-k retrieval.
- View why a chunk was selected and whether the answer is grounded.
- Iterate chunking/retrieval settings and validate improvement.

---

## 3) Dataset Manager

**Purpose:** Prepare and validate JSON datasets for evaluation and batch workflows.

### Core regions
- **Upload panel**
  - JSON file upload
  - Drag-and-drop support (optional)
- **Preview table**
  - Row-level sample view
  - Expandable record details
- **Validation results panel**
  - Schema validation status
  - Field-level validation errors and warnings
  - Row counts (valid vs invalid)

### Key interactions
- Upload dataset JSON.
- Preview records before saving.
- Correct malformed data based on structured error messages.

---

## 4) Batch Evaluation

**Purpose:** Execute reproducible large-scale runs and analyze outcomes.

### Core regions
- **Run configuration form**
  - Dataset selection
  - Prompt template/version
  - Model and parameter settings
  - Concurrency and timeout controls
- **Progress monitor**
  - Overall completion percentage
  - In-flight, succeeded, failed counts
  - ETA and throughput
- **Aggregate metrics panel**
  - Quality and task metrics (accuracy/f1/etc., as applicable)
  - Latency and token aggregates
- **Failure slices panel**
  - Error category breakdown
  - Worst-performing segments (prompt type, input length, source)

### Key interactions
- Configure and launch batch jobs.
- Track run progress in near real time.
- Drill into failures to identify regression patterns.

---

## 5) Metrics Dashboard

**Purpose:** Operational observability of inference usage and reliability.

### Required KPIs
- Requests per minute (RPM)
- Latency percentiles (p50/p95/p99)
- Token usage (input/output/total over time)
- Cache hit rate
- Error rates

### Core regions
- **Time range controls** (last 15m/1h/24h/custom)
- **KPI cards** for at-a-glance health
- **Trend charts** for each required metric
- **Dimension filters** (model, endpoint, tenant, region)

### Key interactions
- Select time windows and filters.
- Compare utilization vs reliability trends.
- Identify spikes in errors, latency, or token consumption.

---

## 6) System Status

**Purpose:** Live system introspection for operators.

### Core regions
- **Model metadata**
  - Loaded models and versions
  - Context window / limits
  - Quantization/runtime details (if applicable)
- **Queue depth panel**
  - Pending request counts
  - Worker utilization/backlog indicators
- **Service health panel**
  - Component-level status (API, retrieval, cache, queue, DB)
  - Last heartbeat and uptime
  - Incident/warning banner area

### Key interactions
- Verify readiness before traffic or evaluation runs.
- Detect bottlenecks (queue buildup, unhealthy dependencies).
- Triage service issues using per-component status.

---

## Global UX expectations

- Consistent tab navigation across all six screens.
- Shared model/environment selector where relevant.
- Cross-screen links (e.g., from Batch Evaluation failures to RAG Explorer evidence).
- Export options for metrics and evaluation artifacts.
- Role-based visibility for sensitive operational views.
