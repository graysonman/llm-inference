import csv
import asyncio
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Dict, Optional, Tuple, List

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    redis = None
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except Exception:
    otel_trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    OTLPSpanExporter = None
    OTEL_AVAILABLE = False

from app.logging_utils import get_logger, log_json
from app.schemas import (
    BatchEvalCreateRequest,
    BatchEvalCreateResponse,
    ChatRequest,
    ChatResponse,
    CriterionScore,
    DatasetCreateRequest,
    DatasetCreateResponse,
    DatasetGetResponse,
    DatasetListItem,
    DatasetListResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EvaluateRequest,
    EvaluateResponse,
    EvalRunSummary,
    RagContractResponse,
    RagIndexStatusResponse,
    RagQueryRequest,
    RagQueryResponse,
    BatchEvalStatusResponse,
    BatchEvalResultResponse,
    BatchEvalFailureItem,
    BatchEvalFailuresResponse,
    BatchEvalDistributionResponse,
)

load_dotenv()

LOGGER = get_logger()
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
AUTH_CONTEXT: ContextVar[Dict[str, Any] | None] = ContextVar("auth_context", default=None)

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("API_KEY", "dev-local-key")
API_KEY_ROLE = os.getenv("API_KEY_ROLE", "admin").strip() or "admin"
API_KEYS_JSON = os.getenv("API_KEYS_JSON", "").strip()
CACHE_TTL_SECONDS = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "120"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "llm-inference:chat:")
STATE_PERSISTENCE_ENABLED = os.getenv("STATE_PERSISTENCE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
STATE_BACKEND = os.getenv("STATE_BACKEND", "json").strip().lower() or "json"
STATE_FILE_PATH = os.getenv(
    "STATE_FILE_PATH",
    os.path.join(tempfile.gettempdir(), "llm-inference", "state.json"),
)
STATE_SQLITE_PATH = os.getenv(
    "STATE_SQLITE_PATH",
    os.path.join(tempfile.gettempdir(), "llm-inference", "state.db"),
)
SOFT_DELETE_RETENTION_SECONDS = int(os.getenv("SOFT_DELETE_RETENTION_SECONDS", "604800"))
RETENTION_SWEEP_ENABLED = os.getenv("RETENTION_SWEEP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
RETENTION_SWEEP_INTERVAL_SECONDS = int(os.getenv("RETENTION_SWEEP_INTERVAL_SECONDS", "300"))
RETENTION_HISTORY_LIMIT = int(os.getenv("RETENTION_HISTORY_LIMIT", "20"))
BATCH_EVENT_HISTORY_LIMIT = int(os.getenv("BATCH_EVENT_HISTORY_LIMIT", "500"))
AUDIT_LOG_MAX_ENTRIES = int(os.getenv("AUDIT_LOG_MAX_ENTRIES", "5000"))
BATCH_EVAL_MAX_RETRIES = int(os.getenv("BATCH_EVAL_MAX_RETRIES", "2"))
BATCH_EVAL_RETRY_BACKOFF_MS = int(os.getenv("BATCH_EVAL_RETRY_BACKOFF_MS", "25"))
BATCH_EVAL_MAX_CONCURRENT_RUNS = int(os.getenv("BATCH_EVAL_MAX_CONCURRENT_RUNS", "2"))
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30"))
CHAT_MAX_CONCURRENT_REQUESTS = int(os.getenv("CHAT_MAX_CONCURRENT_REQUESTS", "8"))
CHAT_REQUEST_TIMEOUT_MS = int(os.getenv("CHAT_REQUEST_TIMEOUT_MS", "30000"))
SLO_WINDOW_SECONDS = int(os.getenv("SLO_WINDOW_SECONDS", "3600"))
SLO_ERROR_BUDGET_PCT = float(os.getenv("SLO_ERROR_BUDGET_PCT", "1.0"))
SLO_INCIDENT_HISTORY_LIMIT = int(os.getenv("SLO_INCIDENT_HISTORY_LIMIT", "200"))
VECTOR_INDEX_BACKEND = (os.getenv("VECTOR_INDEX_BACKEND", "in_memory").strip().lower() or "in_memory")
VECTOR_INDEX_DIM = max(8, int(os.getenv("VECTOR_INDEX_DIM", "16")))
TRACING_ENABLED = os.getenv("TRACING_ENABLED", "0").strip().lower() not in {"0", "false", "no"}
TRACING_OTLP_ENDPOINT = os.getenv("TRACING_OTLP_ENDPOINT", "").strip()

tokenizer = None
model = None
MODEL_META: Dict[str, Any] = {}
CHAT_CACHE: Dict[str, Dict[str, Any]] = {}
METRICS: Dict[str, int] = {
    "chat_requests": 0,
    "chat_cache_hits": 0,
    "chat_cache_misses": 0,
    "chat_backpressure_rejections": 0,
    "chat_timeouts": 0,
    "maintenance_rejections": 0,
}
DATASETS: Dict[str, Dict[str, Any]] = {}
EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
BATCH_EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
RAG_INDEXES: Dict[str, Dict[str, Any]] = {}
RATE_LIMIT_BUCKETS: Dict[str, Dict[str, int]] = {}
AUDIT_LOGS: List[Dict[str, Any]] = []
RUNTIME_CONFIG_PROFILES: Dict[str, Dict[str, Any]] = {}
RUNBOOK_RUNS: Dict[str, Dict[str, Any]] = {}
RUNBOOK_TEMPLATES: Dict[str, Dict[str, Any]] = {}
BATCH_EVAL_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()
CACHE_BACKEND = "in_memory"
REDIS_CLIENT = None
ROLE_SCOPE_DEFAULTS: Dict[str, List[str]] = {
    "admin": ["*"],
    "analyst": [
        "chat:invoke",
        "agent:invoke",
        "evaluate:invoke",
        "embeddings:invoke",
        "rag:query",
        "rag:read",
        "rag:write",
        "datasets:read",
        "datasets:write",
        "evals:read",
        "evals:write",
        "batch:read",
        "batch:write",
        "metrics:read",
    ],
    "viewer": [
        "datasets:read",
        "evals:read",
        "batch:read",
        "rag:read",
        "metrics:read",
    ],
}
AUTH_CAPABILITY_CATALOG: Dict[str, Dict[str, Any]] = {
    "chat.invoke": {"description": "Invoke chat generation", "required_scope": "chat:invoke"},
    "agent.invoke": {"description": "Run operator agent workflows", "required_scope": "agent:invoke"},
    "evaluate.invoke": {"description": "Invoke evaluation endpoint", "required_scope": "evaluate:invoke"},
    "embeddings.invoke": {"description": "Invoke embeddings endpoint", "required_scope": "embeddings:invoke"},
    "rag.query": {"description": "Run retrieval-augmented queries", "required_scope": "rag:query"},
    "rag.read": {"description": "Read RAG indexes and entries", "required_scope": "rag:read"},
    "rag.write": {"description": "Create or mutate RAG indexes", "required_scope": "rag:write"},
    "datasets.read": {"description": "List and read datasets", "required_scope": "datasets:read"},
    "datasets.write": {"description": "Create, upload, or mutate datasets", "required_scope": "datasets:write"},
    "evals.read": {"description": "Read evaluation runs", "required_scope": "evals:read"},
    "evals.write": {"description": "Create evaluation runs", "required_scope": "evals:write"},
    "batch.read": {"description": "Read batch-eval runs and artifacts", "required_scope": "batch:read"},
    "batch.write": {"description": "Create or cancel batch-eval runs", "required_scope": "batch:write"},
    "metrics.read": {"description": "Read metrics endpoints", "required_scope": "metrics:read"},
    "admin.access": {"description": "Access admin operations", "required_role": "admin"},
}
AGENT_TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "datasets.list": {
        "description": "List datasets with optional type/status filters.",
        "required_scope": "datasets:read",
        "params": {
            "limit": "int (optional, default 10)",
            "type": "string (optional: rag_corpus|eval_set)",
            "status": "string (optional: processing|ready|failed)",
        },
    },
    "metrics.dashboard": {
        "description": "Fetch metrics dashboard summary.",
        "required_scope": "metrics:read",
        "params": {
            "window": "string (optional, default 15m)",
        },
    },
    "rag.backend_status": {
        "description": "Inspect RAG vector backend status.",
        "required_scope": "rag:read",
        "params": {},
    },
    "tracing.status": {
        "description": "Inspect tracing backend status.",
        "required_scope": "metrics:read",
        "params": {},
    },
    "runbooks.list": {
        "description": "List admin runbook runs.",
        "required_role": "admin",
        "params": {
            "limit": "int (optional, default 10)",
            "status": "string (optional: in_progress|completed|aborted)",
        },
    },
}
API_KEY_REGISTRY: Dict[str, Dict[str, Any]] = {}
RETENTION_THREAD: threading.Thread | None = None
RETENTION_STOP_EVENT = threading.Event()
RETENTION_STATS: Dict[str, Any] = {
    "last_run_ts": None,
    "last_purged_total": 0,
    "last_error": None,
}
RETENTION_HISTORY: List[Dict[str, Any]] = []
BATCH_QUEUE_SEQ = 0
CIRCUIT_BREAKER_LOCK = threading.Lock()
CHAT_CONCURRENCY_LOCK = threading.Lock()
CHAT_ACTIVE_REQUESTS = 0
RAG_RUNTIME_FAISS_INDEX: Dict[str, Any] = {}
RAG_VECTOR_BACKEND_ACTIVE = "in_memory"
RAG_VECTOR_BACKEND_REASON: Optional[str] = None
TRACING_ACTIVE = False
TRACING_REASON: Optional[str] = None
TRACING_EXPORTER = "none"
TRACER = None
SLO_LOCK = threading.Lock()
SLO_EVENTS: List[Dict[str, Any]] = []
SLO_INCIDENTS: List[Dict[str, Any]] = []
SLO_STATE: Dict[str, Any] = {
    "breached": False,
    "current_incident_id": None,
}
MAINTENANCE_LOCK = threading.Lock()
MAINTENANCE_STATE: Dict[str, Any] = {
    "active": False,
    "reason": None,
    "enabled_at": None,
    "expires_at": None,
    "read_only": False,
}
CIRCUIT_BREAKER: Dict[str, Any] = {
    "state": "closed",  # closed | open | half_open
    "consecutive_failures": 0,
    "opened_at": None,
    "last_failure_at": None,
    "last_failure_reason": None,
    "half_open_trial_inflight": False,
    "manual_forced_open": False,
    "manual_reason": None,
    "manual_expires_at": None,
}


def _cache_backend_name() -> str:
    return CACHE_BACKEND


def _init_cache_backend() -> None:
    global CACHE_BACKEND, REDIS_CLIENT
    CACHE_BACKEND = "in_memory"
    REDIS_CLIENT = None

    if not REDIS_URL:
        return

    if not REDIS_AVAILABLE:
        log_json(
            LOGGER,
            {
                "event": "cache_backend_unavailable",
                "backend": "redis",
                "reason": "redis package not installed",
            },
        )
        return

    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        client.ping()
        REDIS_CLIENT = client
        CACHE_BACKEND = "redis"
        log_json(LOGGER, {"event": "cache_backend_ready", "backend": "redis"})
    except Exception as exc:
        log_json(
            LOGGER,
            {
                "event": "cache_backend_unavailable",
                "backend": "redis",
                "reason": repr(exc),
            },
        )


def _resolve_rag_vector_backend() -> Tuple[str, Optional[str]]:
    configured = VECTOR_INDEX_BACKEND
    if configured not in {"in_memory", "faiss"}:
        return "in_memory", f"unsupported backend '{configured}', falling back to in_memory"
    if configured == "faiss" and (not FAISS_AVAILABLE or not NUMPY_AVAILABLE):
        missing = []
        if not FAISS_AVAILABLE:
            missing.append("faiss")
        if not NUMPY_AVAILABLE:
            missing.append("numpy")
        return "in_memory", f"missing dependencies for faiss backend: {', '.join(missing)}"
    return configured, None


def _init_rag_vector_backend() -> None:
    global RAG_VECTOR_BACKEND_ACTIVE, RAG_VECTOR_BACKEND_REASON, RAG_RUNTIME_FAISS_INDEX
    backend, reason = _resolve_rag_vector_backend()
    RAG_VECTOR_BACKEND_ACTIVE = backend
    RAG_VECTOR_BACKEND_REASON = reason
    RAG_RUNTIME_FAISS_INDEX = {}
    event = {"event": "rag_vector_backend_ready", "configured": VECTOR_INDEX_BACKEND, "active": backend}
    if reason:
        event["reason"] = reason
    log_json(LOGGER, event)


def _rag_vector_backend_snapshot() -> Dict[str, Any]:
    return {
        "configured_backend": VECTOR_INDEX_BACKEND,
        "active_backend": RAG_VECTOR_BACKEND_ACTIVE,
        "vector_dim": VECTOR_INDEX_DIM,
        "faiss_available": bool(FAISS_AVAILABLE),
        "numpy_available": bool(NUMPY_AVAILABLE),
        "fallback_reason": RAG_VECTOR_BACKEND_REASON,
    }


def _record_to_rag_text(record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return str(record)
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    input_value = record.get("input")
    if isinstance(input_value, dict):
        prompt = input_value.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        return json.dumps(input_value, separators=(",", ":"), ensure_ascii=True)
    if isinstance(input_value, str) and input_value.strip():
        return input_value.strip()
    return json.dumps(record, separators=(",", ":"), ensure_ascii=True)


def _text_to_embedding(text: str, dim: int = VECTOR_INDEX_DIM) -> List[float]:
    values = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dim):
            values[i] += ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-9:
        return values
    return [v / norm for v in values]


def _build_rag_chunks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        text = _record_to_rag_text(record)
        chunks.append(
            {
                "chunk_id": f"chunk_{idx + 1}",
                "text": text,
                "vector": _text_to_embedding(text),
            }
        )
    return chunks


def _upsert_rag_index(dataset_id: str, records: List[Dict[str, Any]], now: int | None = None) -> None:
    ts = _now_ts() if now is None else int(now)
    chunks = _build_rag_chunks(records)
    RAG_INDEXES[dataset_id] = {
        "index_id": dataset_id,
        "dataset_id": dataset_id,
        "status": "ready",
        "chunk_count": len(chunks),
        "updated_at": ts,
        "deleted_at": None,
        "backend": RAG_VECTOR_BACKEND_ACTIVE,
        "vector_dim": VECTOR_INDEX_DIM,
        "chunks": chunks,
    }
    _persist_rag_index_record(RAG_INDEXES[dataset_id])
    if dataset_id in RAG_RUNTIME_FAISS_INDEX:
        RAG_RUNTIME_FAISS_INDEX.pop(dataset_id, None)


def _score_rag_chunk(query_vector: List[float], query_tokens: set[str], chunk: Dict[str, Any]) -> float:
    chunk_vector = chunk.get("vector") if isinstance(chunk.get("vector"), list) else []
    dot = 0.0
    if chunk_vector and len(chunk_vector) == len(query_vector):
        dot = float(sum(float(a) * float(b) for a, b in zip(query_vector, chunk_vector)))
    chunk_text = str(chunk.get("text", "")).lower()
    lexical_bonus = 0.0
    if query_tokens:
        overlap = sum(1 for token in query_tokens if token in chunk_text)
        lexical_bonus = min(0.35, 0.08 * overlap)
    return dot + lexical_bonus


def _search_rag_chunks(index_obj: Dict[str, Any], query: str, top_k: int) -> List[Dict[str, Any]]:
    chunks = index_obj.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        return []

    query_vector = _text_to_embedding(query)
    query_tokens = {t for t in query.lower().split() if t}
    clamped_top_k = max(1, min(int(top_k), 50))
    backend = str(index_obj.get("backend", RAG_VECTOR_BACKEND_ACTIVE))

    rows: List[Tuple[int, float]] = []
    if backend == "faiss" and FAISS_AVAILABLE and NUMPY_AVAILABLE:
        cache_key = str(index_obj.get("index_id", ""))
        runtime = RAG_RUNTIME_FAISS_INDEX.get(cache_key)
        needs_rebuild = True
        if isinstance(runtime, dict):
            needs_rebuild = int(runtime.get("updated_at", 0) or 0) != int(index_obj.get("updated_at", 0) or 0)
        if needs_rebuild:
            matrix = [chunk.get("vector", []) for chunk in chunks if isinstance(chunk, dict)]
            if matrix and all(isinstance(v, list) and len(v) == VECTOR_INDEX_DIM for v in matrix):
                np_matrix = np.array(matrix, dtype="float32")
                faiss_index = faiss.IndexFlatIP(VECTOR_INDEX_DIM)
                faiss_index.add(np_matrix)
                RAG_RUNTIME_FAISS_INDEX[cache_key] = {"updated_at": int(index_obj.get("updated_at", 0) or 0), "index": faiss_index}
                runtime = RAG_RUNTIME_FAISS_INDEX[cache_key]
        faiss_index = runtime.get("index") if isinstance(runtime, dict) else None
        if faiss_index is not None:
            q = np.array([query_vector], dtype="float32")
            scores, ids = faiss_index.search(q, min(clamped_top_k, len(chunks)))
            for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
                if idx < 0:
                    continue
                bonus = _score_rag_chunk(query_vector, query_tokens, chunks[idx]) - float(sum(float(a) * float(b) for a, b in zip(query_vector, chunks[idx].get("vector", []))))
                rows.append((int(idx), float(score) + bonus))

    if not rows:
        for idx, chunk in enumerate(chunks):
            rows.append((idx, _score_rag_chunk(query_vector, query_tokens, chunk)))

    rows = sorted(rows, key=lambda item: item[1], reverse=True)[:clamped_top_k]
    out = []
    for idx, score in rows:
        chunk = chunks[idx]
        out.append(
            {
                "chunk_id": str(chunk.get("chunk_id", f"chunk_{idx + 1}")),
                "dataset_id": str(index_obj.get("dataset_id", "")),
                "score": round(float(score), 6),
                "text": str(chunk.get("text", "")),
            }
        )
    return out


def _init_tracing_backend() -> None:
    global TRACING_ACTIVE, TRACING_REASON, TRACING_EXPORTER, TRACER
    TRACING_ACTIVE = False
    TRACING_REASON = None
    TRACING_EXPORTER = "none"
    TRACER = None

    if not TRACING_ENABLED:
        TRACING_REASON = "disabled by configuration"
        log_json(LOGGER, {"event": "tracing_backend_ready", "active": False, "reason": TRACING_REASON})
        return

    if not OTEL_AVAILABLE:
        TRACING_REASON = "opentelemetry dependencies are not installed"
        log_json(LOGGER, {"event": "tracing_backend_ready", "active": False, "reason": TRACING_REASON})
        return

    if not TRACING_OTLP_ENDPOINT:
        TRACING_REASON = "TRACING_OTLP_ENDPOINT is not configured"
        log_json(LOGGER, {"event": "tracing_backend_ready", "active": False, "reason": TRACING_REASON})
        return

    try:
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=TRACING_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)
        TRACER = otel_trace.get_tracer("llm-inference")
        TRACING_ACTIVE = True
        TRACING_EXPORTER = "otlp_http"
        TRACING_REASON = None
        log_json(
            LOGGER,
            {
                "event": "tracing_backend_ready",
                "active": True,
                "exporter": TRACING_EXPORTER,
                "endpoint": TRACING_OTLP_ENDPOINT,
            },
        )
    except Exception as exc:
        TRACING_ACTIVE = False
        TRACING_REASON = repr(exc)
        TRACING_EXPORTER = "none"
        TRACER = None
        log_json(LOGGER, {"event": "tracing_backend_ready", "active": False, "reason": TRACING_REASON})


def _tracing_backend_snapshot() -> Dict[str, Any]:
    return {
        "enabled": bool(TRACING_ENABLED),
        "active": bool(TRACING_ACTIVE),
        "exporter": TRACING_EXPORTER,
        "otlp_endpoint": TRACING_OTLP_ENDPOINT or None,
        "otel_available": bool(OTEL_AVAILABLE),
        "reason": TRACING_REASON,
    }


def _role_scopes(role: str) -> List[str]:
    normalized = (role or "").strip().lower()
    return list(ROLE_SCOPE_DEFAULTS.get(normalized, []))


def _build_api_key_registry(default_key: str, default_role: str, raw_json: str) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {
        default_key: {"role": default_role, "scopes": _role_scopes(default_role)},
    }

    if not raw_json:
        return registry

    try:
        loaded = json.loads(raw_json)
    except Exception:
        log_json(LOGGER, {"event": "auth_config_invalid_json"})
        return registry

    if not isinstance(loaded, dict):
        log_json(LOGGER, {"event": "auth_config_invalid_shape", "detail": "API_KEYS_JSON must be an object"})
        return registry

    for key, value in loaded.items():
        if not isinstance(key, str) or not key.strip():
            continue
        role = "viewer"
        scopes: List[str] = []

        if isinstance(value, str):
            role = value.strip().lower() or "viewer"
            scopes = _role_scopes(role)
        elif isinstance(value, list):
            scopes = [str(v).strip() for v in value if str(v).strip()]
        elif isinstance(value, dict):
            role = str(value.get("role", "viewer")).strip().lower() or "viewer"
            raw_scopes = value.get("scopes")
            if isinstance(raw_scopes, list):
                scopes = [str(v).strip() for v in raw_scopes if str(v).strip()]
            else:
                scopes = _role_scopes(role)
        else:
            continue

        if not scopes:
            scopes = _role_scopes(role)

        registry[key] = {"role": role, "scopes": scopes}

    return registry


def _init_auth_registry() -> None:
    global API_KEY_REGISTRY
    API_KEY_REGISTRY = _build_api_key_registry(API_KEY, API_KEY_ROLE, API_KEYS_JSON)


def _authenticate_api_key_value(provided: Optional[str]) -> Optional[Dict[str, Any]]:
    if not provided:
        return None
    record = API_KEY_REGISTRY.get(provided)
    if record is None:
        return None
    return {
        "api_key": provided,
        "role": str(record.get("role", "viewer")),
        "scopes": [str(s) for s in record.get("scopes", [])],
    }


def _scope_allowed(granted_scopes: List[str], required_scope: str) -> bool:
    normalized_granted = [s.strip().lower() for s in granted_scopes]
    req = required_scope.strip().lower()
    if "*" in normalized_granted or req in normalized_granted:
        return True
    if ":" in req:
        prefix = req.split(":", 1)[0]
        if f"{prefix}:*" in normalized_granted:
            return True
    return False


def _build_auth_capabilities(auth: Dict[str, Any]) -> Dict[str, bool]:
    role = str(auth.get("role", "viewer")).strip().lower()
    scopes = [str(s).strip() for s in auth.get("scopes", []) if str(s).strip()]

    def allows(scope: str) -> bool:
        return _scope_allowed(scopes, scope)

    return {
        "chat.invoke": allows("chat:invoke"),
        "agent.invoke": allows("agent:invoke"),
        "evaluate.invoke": allows("evaluate:invoke"),
        "embeddings.invoke": allows("embeddings:invoke"),
        "rag.query": allows("rag:query"),
        "rag.read": allows("rag:read"),
        "rag.write": allows("rag:write"),
        "datasets.read": allows("datasets:read"),
        "datasets.write": allows("datasets:write"),
        "evals.read": allows("evals:read"),
        "evals.write": allows("evals:write"),
        "batch.read": allows("batch:read"),
        "batch.write": allows("batch:write"),
        "metrics.read": allows("metrics:read"),
        "admin.access": role == "admin",
    }


def _require_auth_for_scopes(
    required_scopes: List[str],
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Dict[str, Any]:
    provided = _resolve_api_key(x_api_key, authorization)
    auth = _authenticate_api_key_value(provided)
    if auth is None:
        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API key"})

    missing = [scope for scope in required_scopes if not _scope_allowed(auth["scopes"], scope)]
    if missing:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "forbidden",
                "message": "Insufficient scope",
                "details": {
                    "required_scopes": required_scopes,
                    "missing_scopes": missing,
                    "role": auth["role"],
                },
            },
        )

    AUTH_CONTEXT.set(auth)
    return auth


def _authenticate_from_headers(x_api_key: str | None, authorization: str | None) -> Dict[str, Any]:
    direct_call = not isinstance(x_api_key, (str, type(None))) and not isinstance(authorization, (str, type(None)))
    normalized_x_api_key = x_api_key if isinstance(x_api_key, str) else None
    normalized_authorization = authorization if isinstance(authorization, str) else None

    provided = _resolve_api_key(normalized_x_api_key, normalized_authorization)
    auth = _authenticate_api_key_value(provided)
    if auth is not None:
        return auth

    if direct_call:
        fallback = AUTH_CONTEXT.get()
        if isinstance(fallback, dict):
            return dict(fallback)
        return {}

    raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API key"})


def _state_backend_name() -> str:
    if not STATE_PERSISTENCE_ENABLED:
        return "in_memory"
    if STATE_BACKEND == "sqlite":
        return f"sqlite:{STATE_SQLITE_PATH}"
    return f"file_json:{STATE_FILE_PATH}"


def _build_state_snapshot() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": _now_ts(),
        "state_backend": _state_backend_name(),
        "data": {
            "datasets": DATASETS,
            "eval_runs": EVAL_RUNS,
            "batch_eval_runs": BATCH_EVAL_RUNS,
            "rag_indexes": RAG_INDEXES,
            "audit_logs": AUDIT_LOGS,
            "runtime_config_profiles": RUNTIME_CONFIG_PROFILES,
            "runbook_runs": RUNBOOK_RUNS,
            "runbook_templates": RUNBOOK_TEMPLATES,
            "maintenance_state": MAINTENANCE_STATE,
            "slo_incidents": SLO_INCIDENTS,
            "slo_state": SLO_STATE,
        },
        "retention": {
            "stats": RETENTION_STATS,
            "history": RETENTION_HISTORY,
        },
    }


def _extract_snapshot_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "payload must be an object"})

    source = payload.get("data", payload)
    if not isinstance(source, dict):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "data must be an object"})

    return {
        "datasets": source.get("datasets", {}),
        "eval_runs": source.get("eval_runs", {}),
        "batch_eval_runs": source.get("batch_eval_runs", {}),
        "rag_indexes": source.get("rag_indexes", {}),
        "audit_logs": source.get("audit_logs", []),
        "runtime_config_profiles": source.get("runtime_config_profiles", {}),
        "runbook_runs": source.get("runbook_runs", {}),
        "runbook_templates": source.get("runbook_templates", {}),
        "maintenance_state": source.get("maintenance_state", {}),
        "slo_incidents": source.get("slo_incidents", []),
        "slo_state": source.get("slo_state", {}),
    }


def _audit_key_fingerprint(api_key: Optional[str]) -> str:
    if not api_key:
        return "anonymous"
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def _trim_audit_logs() -> None:
    max_len = max(100, AUDIT_LOG_MAX_ENTRIES)
    if len(AUDIT_LOGS) > max_len:
        del AUDIT_LOGS[:-max_len]


def _normalize_loaded_state(
    datasets: Any,
    eval_runs: Any,
    batch_eval_runs: Any,
    rag_indexes: Any,
    audit_logs: Any,
    runtime_config_profiles: Any,
    runbook_runs: Any,
    runbook_templates: Any,
    maintenance_state: Any,
    slo_incidents: Any,
    slo_state: Any,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    if not isinstance(datasets, dict):
        datasets = {}
    if not isinstance(eval_runs, dict):
        eval_runs = {}
    if not isinstance(batch_eval_runs, dict):
        batch_eval_runs = {}
    if not isinstance(rag_indexes, dict):
        rag_indexes = {}
    if not isinstance(audit_logs, list):
        audit_logs = []
    if not isinstance(runtime_config_profiles, dict):
        runtime_config_profiles = {}
    if not isinstance(runbook_runs, dict):
        runbook_runs = {}
    if not isinstance(runbook_templates, dict):
        runbook_templates = {}
    if not isinstance(maintenance_state, dict):
        maintenance_state = {}
    if not isinstance(slo_incidents, list):
        slo_incidents = []
    if not isinstance(slo_state, dict):
        slo_state = {}

    now = _now_ts()
    for run in batch_eval_runs.values():
        if not isinstance(run, dict):
            continue
        if run.get("status") in {"queued", "running"}:
            run["status"] = "failed"
            run["updated_at"] = now
            run["completed_at"] = now
            summary = run.get("summary")
            if not isinstance(summary, dict):
                summary = {}
            summary["error"] = "batch interrupted by service restart"
            run["summary"] = summary

    normalized_audit = [x for x in audit_logs if isinstance(x, dict)]
    normalized_profiles: Dict[str, Dict[str, Any]] = {}
    for name, cfg in runtime_config_profiles.items():
        if not isinstance(name, str):
            continue
        key = name.strip()
        if not key or len(key) > 120:
            continue
        if not isinstance(cfg, dict):
            continue
        normalized_profiles[key] = dict(cfg)

    normalized_runbooks: Dict[str, Dict[str, Any]] = {}
    for runbook_id, payload in runbook_runs.items():
        if not isinstance(runbook_id, str):
            continue
        key = runbook_id.strip()
        if not key:
            continue
        if not isinstance(payload, dict):
            continue
        normalized_runbooks[key] = dict(payload)
    normalized_templates: Dict[str, Dict[str, Any]] = {}
    for template_id, payload in runbook_templates.items():
        if not isinstance(template_id, str):
            continue
        key = template_id.strip()
        if not key:
            continue
        if not isinstance(payload, dict):
            continue
        normalized_templates[key] = dict(payload)
    normalized_maintenance: Dict[str, Any] = {
        "active": bool(maintenance_state.get("active", False)),
        "reason": maintenance_state.get("reason"),
        "enabled_at": maintenance_state.get("enabled_at"),
        "expires_at": maintenance_state.get("expires_at"),
        "read_only": bool(maintenance_state.get("read_only", False)),
    }
    normalized_incidents: List[Dict[str, Any]] = []
    for item in slo_incidents:
        if not isinstance(item, dict):
            continue
        normalized_incidents.append(dict(item))
    max_incidents = max(10, int(SLO_INCIDENT_HISTORY_LIMIT))
    if len(normalized_incidents) > max_incidents:
        normalized_incidents = normalized_incidents[-max_incidents:]

    normalized_slo_state: Dict[str, Any] = {
        "breached": bool(slo_state.get("breached", False)),
        "current_incident_id": slo_state.get("current_incident_id"),
    }
    return (
        datasets,
        eval_runs,
        batch_eval_runs,
        rag_indexes,
        normalized_audit,
        normalized_profiles,
        normalized_runbooks,
        normalized_templates,
        normalized_maintenance,
        normalized_incidents,
        normalized_slo_state,
    )


def _save_state_json() -> None:
    state = {
        "datasets": DATASETS,
        "eval_runs": EVAL_RUNS,
        "batch_eval_runs": BATCH_EVAL_RUNS,
        "rag_indexes": RAG_INDEXES,
        "audit_logs": AUDIT_LOGS,
        "runtime_config_profiles": RUNTIME_CONFIG_PROFILES,
        "runbook_runs": RUNBOOK_RUNS,
        "runbook_templates": RUNBOOK_TEMPLATES,
        "maintenance_state": MAINTENANCE_STATE,
        "slo_incidents": SLO_INCIDENTS,
        "slo_state": SLO_STATE,
    }

    parent = os.path.dirname(STATE_FILE_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
    tmp_path = f"{STATE_FILE_PATH}.tmp"
    with STATE_LOCK:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, STATE_FILE_PATH)


def _load_state_json() -> None:
    if not os.path.exists(STATE_FILE_PATH):
        return

    with open(STATE_FILE_PATH, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)

    if not isinstance(loaded, dict):
        return

    datasets, eval_runs, batch_eval_runs, rag_indexes, audit_logs, runtime_config_profiles, runbook_runs, runbook_templates, maintenance_state, slo_incidents, slo_state = _normalize_loaded_state(
        loaded.get("datasets", {}),
        loaded.get("eval_runs", {}),
        loaded.get("batch_eval_runs", {}),
        loaded.get("rag_indexes", {}),
        loaded.get("audit_logs", []),
        loaded.get("runtime_config_profiles", {}),
        loaded.get("runbook_runs", {}),
        loaded.get("runbook_templates", {}),
        loaded.get("maintenance_state", {}),
        loaded.get("slo_incidents", []),
        loaded.get("slo_state", {}),
    )

    with STATE_LOCK:
        DATASETS.clear()
        DATASETS.update(datasets)
        EVAL_RUNS.clear()
        EVAL_RUNS.update(eval_runs)
        BATCH_EVAL_RUNS.clear()
        BATCH_EVAL_RUNS.update(batch_eval_runs)
        RAG_INDEXES.clear()
        RAG_INDEXES.update(rag_indexes)
        AUDIT_LOGS.clear()
        AUDIT_LOGS.extend(audit_logs)
        RUNTIME_CONFIG_PROFILES.clear()
        RUNTIME_CONFIG_PROFILES.update(runtime_config_profiles)
        RUNBOOK_RUNS.clear()
        RUNBOOK_RUNS.update(runbook_runs)
        RUNBOOK_TEMPLATES.clear()
        RUNBOOK_TEMPLATES.update(runbook_templates)
        MAINTENANCE_STATE.clear()
        MAINTENANCE_STATE.update(maintenance_state)
        SLO_INCIDENTS.clear()
        SLO_INCIDENTS.extend(slo_incidents)
        SLO_STATE.clear()
        SLO_STATE.update(slo_state)
        _trim_audit_logs()


def _ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_eval_runs (
            run_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_indexes (
            index_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            event_id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            actor_role TEXT,
            actor_key TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            request_id TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_config_profiles (
            profile_name TEXT PRIMARY KEY,
            updated_at INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runbook_runs (
            runbook_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runbook_templates (
            template_id TEXT PRIMARY KEY,
            updated_at INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS service_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slo_incidents (
            incident_id TEXT PRIMARY KEY,
            opened_at INTEGER NOT NULL,
            status TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )

    desired_columns: Dict[str, Dict[str, str]] = {
        "datasets": {
            "name": "TEXT",
            "type": "TEXT",
            "status": "TEXT",
            "record_count": "INTEGER",
            "created_at": "INTEGER",
            "error": "TEXT",
            "deleted_at": "INTEGER",
        },
        "eval_runs": {
            "model": "TEXT",
            "latency_ms": "INTEGER",
            "created_at": "INTEGER",
            "deleted_at": "INTEGER",
        },
        "batch_eval_runs": {
            "batch_eval_id": "TEXT",
            "dataset_id": "TEXT",
            "status": "TEXT",
            "created_at": "INTEGER",
            "started_at": "INTEGER",
            "completed_at": "INTEGER",
            "deleted_at": "INTEGER",
        },
        "rag_indexes": {
            "dataset_id": "TEXT",
            "status": "TEXT",
            "chunk_count": "INTEGER",
            "deleted_at": "INTEGER",
        },
    }

    for table, cols in desired_columns.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, col_type in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_type_status_created ON datasets(type, status, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_runs_status_created ON batch_eval_runs(status, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_indexes_dataset_status ON rag_indexes(dataset_id, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_ts ON audit_logs(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_action_ts ON audit_logs(action, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_ts ON audit_logs(resource_type, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_profiles_updated ON runtime_config_profiles(updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runbook_runs_status_created ON runbook_runs(status, created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runbook_templates_updated ON runbook_templates(updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_slo_incidents_opened ON slo_incidents(opened_at DESC)")


def _save_state_sqlite() -> None:
    parent = os.path.dirname(STATE_SQLITE_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            now = _now_ts()
            conn.execute("DELETE FROM datasets")
            conn.execute("DELETE FROM eval_runs")
            conn.execute("DELETE FROM batch_eval_runs")
            conn.execute("DELETE FROM rag_indexes")
            conn.execute("DELETE FROM audit_logs")
            conn.execute("DELETE FROM runtime_config_profiles")
            conn.execute("DELETE FROM runbook_runs")
            conn.execute("DELETE FROM runbook_templates")
            conn.execute("DELETE FROM service_state")
            conn.execute("DELETE FROM slo_incidents")
            for dataset_id, payload in DATASETS.items():
                conn.execute(
                    """
                    INSERT INTO datasets(dataset_id, name, type, status, record_count, created_at, updated_at, error, deleted_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_id) DO UPDATE SET
                        name=excluded.name,
                        type=excluded.type,
                        status=excluded.status,
                        record_count=excluded.record_count,
                        created_at=excluded.created_at,
                        error=excluded.error,
                        deleted_at=excluded.deleted_at,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        dataset_id,
                        str(payload.get("name", "")),
                        str(payload.get("type", "")),
                        str(payload.get("status", "")),
                        int(payload.get("record_count", 0) or 0),
                        int(payload.get("created_at", now) or now),
                        int(payload.get("updated_at", now) or now),
                        payload.get("error"),
                        payload.get("deleted_at"),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for run_id, payload in EVAL_RUNS.items():
                conn.execute(
                    """
                    INSERT INTO eval_runs(run_id, model, latency_ms, created_at, updated_at, deleted_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        model=excluded.model,
                        latency_ms=excluded.latency_ms,
                        created_at=excluded.created_at,
                        deleted_at=excluded.deleted_at,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        str(payload.get("model", "")),
                        int(payload.get("latency_ms", 0) or 0),
                        int(payload.get("created_at", now) or now),
                        now,
                        payload.get("deleted_at"),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for run_id, payload in BATCH_EVAL_RUNS.items():
                conn.execute(
                    """
                    INSERT INTO batch_eval_runs(run_id, batch_eval_id, dataset_id, status, created_at, started_at, completed_at, updated_at, deleted_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        batch_eval_id=excluded.batch_eval_id,
                        dataset_id=excluded.dataset_id,
                        status=excluded.status,
                        created_at=excluded.created_at,
                        started_at=excluded.started_at,
                        completed_at=excluded.completed_at,
                        deleted_at=excluded.deleted_at,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        str(payload.get("batch_eval_id", run_id)),
                        str(payload.get("dataset_id", "")),
                        str(payload.get("status", "")),
                        int(payload.get("created_at", now) or now),
                        payload.get("started_at"),
                        payload.get("completed_at"),
                        int(payload.get("updated_at", now) or now),
                        payload.get("deleted_at"),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for index_id, payload in RAG_INDEXES.items():
                conn.execute(
                    """
                    INSERT INTO rag_indexes(index_id, dataset_id, status, chunk_count, updated_at, deleted_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(index_id) DO UPDATE SET
                        dataset_id=excluded.dataset_id,
                        status=excluded.status,
                        chunk_count=excluded.chunk_count,
                        deleted_at=excluded.deleted_at,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        index_id,
                        str(payload.get("dataset_id", "")),
                        str(payload.get("status", "")),
                        int(payload.get("chunk_count", 0) or 0),
                        int(payload.get("updated_at", now) or now),
                        payload.get("deleted_at"),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for event in AUDIT_LOGS:
                conn.execute(
                    """
                    INSERT INTO audit_logs(event_id, ts, actor_role, actor_key, action, resource_type, resource_id, request_id, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        ts=excluded.ts,
                        actor_role=excluded.actor_role,
                        actor_key=excluded.actor_key,
                        action=excluded.action,
                        resource_type=excluded.resource_type,
                        resource_id=excluded.resource_id,
                        request_id=excluded.request_id,
                        payload=excluded.payload
                    """,
                    (
                        str(event.get("event_id", "")),
                        int(event.get("ts", now) or now),
                        event.get("actor_role"),
                        event.get("actor_key"),
                        str(event.get("action", "")),
                        str(event.get("resource_type", "")),
                        event.get("resource_id"),
                        event.get("request_id"),
                        json.dumps(event, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for profile_name, payload in RUNTIME_CONFIG_PROFILES.items():
                conn.execute(
                    """
                    INSERT INTO runtime_config_profiles(profile_name, updated_at, payload)
                    VALUES (?, ?, ?)
                    ON CONFLICT(profile_name) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                    """,
                    (
                        str(profile_name),
                        int(payload.get("updated_at", now) or now),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for runbook_id, payload in RUNBOOK_RUNS.items():
                conn.execute(
                    """
                    INSERT INTO runbook_runs(runbook_id, status, created_at, updated_at, payload)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(runbook_id) DO UPDATE SET
                        status=excluded.status,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                    """,
                    (
                        str(runbook_id),
                        str(payload.get("status", "in_progress")),
                        int(payload.get("created_at", now) or now),
                        int(payload.get("updated_at", now) or now),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            for template_id, payload in RUNBOOK_TEMPLATES.items():
                conn.execute(
                    """
                    INSERT INTO runbook_templates(template_id, updated_at, payload)
                    VALUES (?, ?, ?)
                    ON CONFLICT(template_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                    """,
                    (
                        str(template_id),
                        int(payload.get("updated_at", now) or now),
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            conn.execute(
                """
                INSERT INTO service_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (
                    "maintenance_state",
                    json.dumps(MAINTENANCE_STATE, separators=(",", ":"), ensure_ascii=True),
                    now,
                ),
            )
            for incident in SLO_INCIDENTS:
                conn.execute(
                    """
                    INSERT INTO slo_incidents(incident_id, opened_at, status, updated_at, payload)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(incident_id) DO UPDATE SET
                        opened_at=excluded.opened_at,
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                    """,
                    (
                        str(incident.get("incident_id", "")),
                        int(incident.get("opened_at", now) or now),
                        str(incident.get("status", "resolved")),
                        int(incident.get("updated_at", now) or now),
                        json.dumps(incident, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
            conn.execute(
                """
                INSERT INTO service_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (
                    "slo_state",
                    json.dumps(SLO_STATE, separators=(",", ":"), ensure_ascii=True),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _load_state_sqlite() -> None:
    if not os.path.exists(STATE_SQLITE_PATH):
        return

    conn = sqlite3.connect(STATE_SQLITE_PATH)
    try:
        _ensure_sqlite_schema(conn)
        dataset_rows = conn.execute("SELECT dataset_id, payload FROM datasets").fetchall()
        eval_rows = conn.execute("SELECT run_id, payload FROM eval_runs").fetchall()
        batch_rows = conn.execute("SELECT run_id, payload FROM batch_eval_runs").fetchall()
        rag_rows = conn.execute("SELECT index_id, payload FROM rag_indexes").fetchall()
        audit_rows = conn.execute("SELECT event_id, payload FROM audit_logs ORDER BY ts DESC").fetchall()
        profile_rows = conn.execute("SELECT profile_name, payload FROM runtime_config_profiles").fetchall()
        runbook_rows = conn.execute("SELECT runbook_id, payload FROM runbook_runs").fetchall()
        template_rows = conn.execute("SELECT template_id, payload FROM runbook_templates").fetchall()
        service_rows = conn.execute("SELECT key, value FROM service_state").fetchall()
        slo_incident_rows = conn.execute("SELECT incident_id, payload FROM slo_incidents ORDER BY opened_at DESC").fetchall()
        legacy_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='state_kv'"
        ).fetchall()
        state_kv_rows = []
        if legacy_rows:
            state_kv_rows = conn.execute("SELECT key, value FROM state_kv").fetchall()
    finally:
        conn.close()

    datasets_loaded: Dict[str, Any] = {}
    eval_runs_loaded: Dict[str, Any] = {}
    batch_eval_runs_loaded: Dict[str, Any] = {}
    rag_indexes_loaded: Dict[str, Any] = {}
    audit_logs_loaded: List[Dict[str, Any]] = []
    runtime_profiles_loaded: Dict[str, Dict[str, Any]] = {}
    runbook_runs_loaded: Dict[str, Dict[str, Any]] = {}
    runbook_templates_loaded: Dict[str, Dict[str, Any]] = {}
    maintenance_state_loaded: Dict[str, Any] = {}
    slo_incidents_loaded: List[Dict[str, Any]] = []
    slo_state_loaded: Dict[str, Any] = {}

    for dataset_id, payload in dataset_rows:
        try:
            datasets_loaded[str(dataset_id)] = json.loads(payload)
        except Exception:
            continue
    for run_id, payload in eval_rows:
        try:
            eval_runs_loaded[str(run_id)] = json.loads(payload)
        except Exception:
            continue
    for run_id, payload in batch_rows:
        try:
            batch_eval_runs_loaded[str(run_id)] = json.loads(payload)
        except Exception:
            continue
    for index_id, payload in rag_rows:
        try:
            rag_indexes_loaded[str(index_id)] = json.loads(payload)
        except Exception:
            continue
    for _, payload in audit_rows:
        try:
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                audit_logs_loaded.append(loaded)
        except Exception:
            continue
    for profile_name, payload in profile_rows:
        try:
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                runtime_profiles_loaded[str(profile_name)] = loaded
        except Exception:
            continue
    for runbook_id, payload in runbook_rows:
        try:
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                runbook_runs_loaded[str(runbook_id)] = loaded
        except Exception:
            continue
    for template_id, payload in template_rows:
        try:
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                runbook_templates_loaded[str(template_id)] = loaded
        except Exception:
            continue
    for key, value in service_rows:
        k = str(key)
        try:
            loaded = json.loads(value)
            if isinstance(loaded, dict):
                if k == "maintenance_state":
                    maintenance_state_loaded = loaded
                elif k == "slo_state":
                    slo_state_loaded = loaded
        except Exception:
            continue
    for _, payload in slo_incident_rows:
        try:
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                slo_incidents_loaded.append(loaded)
        except Exception:
            continue

    # Backward compatibility for older SQLite files that used a single state_kv blob table.
    legacy_loaded_map: Dict[str, Any] = {}
    for key, value in state_kv_rows:
        try:
            legacy_loaded_map[str(key)] = json.loads(value)
        except Exception:
            legacy_loaded_map[str(key)] = {}

    if not datasets_loaded:
        maybe = legacy_loaded_map.get("datasets", {})
        if isinstance(maybe, dict):
            datasets_loaded = maybe
    if not eval_runs_loaded:
        maybe = legacy_loaded_map.get("eval_runs", {})
        if isinstance(maybe, dict):
            eval_runs_loaded = maybe
    if not batch_eval_runs_loaded:
        maybe = legacy_loaded_map.get("batch_eval_runs", {})
        if isinstance(maybe, dict):
            batch_eval_runs_loaded = maybe
    if not rag_indexes_loaded:
        maybe = legacy_loaded_map.get("rag_indexes", {})
        if isinstance(maybe, dict):
            rag_indexes_loaded = maybe
    if not audit_logs_loaded:
        maybe = legacy_loaded_map.get("audit_logs", [])
        if isinstance(maybe, list):
            audit_logs_loaded = [x for x in maybe if isinstance(x, dict)]
    if not runtime_profiles_loaded:
        maybe = legacy_loaded_map.get("runtime_config_profiles", {})
        if isinstance(maybe, dict):
            runtime_profiles_loaded = {str(k): v for k, v in maybe.items() if isinstance(v, dict)}
    if not runbook_runs_loaded:
        maybe = legacy_loaded_map.get("runbook_runs", {})
        if isinstance(maybe, dict):
            runbook_runs_loaded = {str(k): v for k, v in maybe.items() if isinstance(v, dict)}
    if not runbook_templates_loaded:
        maybe = legacy_loaded_map.get("runbook_templates", {})
        if isinstance(maybe, dict):
            runbook_templates_loaded = {str(k): v for k, v in maybe.items() if isinstance(v, dict)}
    if not maintenance_state_loaded:
        maybe = legacy_loaded_map.get("maintenance_state", {})
        if isinstance(maybe, dict):
            maintenance_state_loaded = maybe
    if not slo_incidents_loaded:
        maybe = legacy_loaded_map.get("slo_incidents", [])
        if isinstance(maybe, list):
            slo_incidents_loaded = [x for x in maybe if isinstance(x, dict)]
    if not slo_state_loaded:
        maybe = legacy_loaded_map.get("slo_state", {})
        if isinstance(maybe, dict):
            slo_state_loaded = maybe

    datasets, eval_runs, batch_eval_runs, rag_indexes, audit_logs, runtime_profiles, runbook_runs, runbook_templates, maintenance_state, slo_incidents, slo_state = _normalize_loaded_state(
        datasets_loaded,
        eval_runs_loaded,
        batch_eval_runs_loaded,
        rag_indexes_loaded,
        audit_logs_loaded,
        runtime_profiles_loaded,
        runbook_runs_loaded,
        runbook_templates_loaded,
        maintenance_state_loaded,
        slo_incidents_loaded,
        slo_state_loaded,
    )

    with STATE_LOCK:
        DATASETS.clear()
        DATASETS.update(datasets)
        EVAL_RUNS.clear()
        EVAL_RUNS.update(eval_runs)
        BATCH_EVAL_RUNS.clear()
        BATCH_EVAL_RUNS.update(batch_eval_runs)
        RAG_INDEXES.clear()
        RAG_INDEXES.update(rag_indexes)
        AUDIT_LOGS.clear()
        AUDIT_LOGS.extend(audit_logs)
        RUNTIME_CONFIG_PROFILES.clear()
        RUNTIME_CONFIG_PROFILES.update(runtime_profiles)
        RUNBOOK_RUNS.clear()
        RUNBOOK_RUNS.update(runbook_runs)
        RUNBOOK_TEMPLATES.clear()
        RUNBOOK_TEMPLATES.update(runbook_templates)
        MAINTENANCE_STATE.clear()
        MAINTENANCE_STATE.update(maintenance_state)
        SLO_INCIDENTS.clear()
        SLO_INCIDENTS.extend(slo_incidents)
        SLO_STATE.clear()
        SLO_STATE.update(slo_state)
        _trim_audit_logs()


def _sqlite_upsert_dataset_record(dataset: Dict[str, Any]) -> None:
    now = _now_ts()
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            conn.execute(
                """
                INSERT INTO datasets(dataset_id, name, type, status, record_count, created_at, updated_at, error, deleted_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name=excluded.name,
                    type=excluded.type,
                    status=excluded.status,
                    record_count=excluded.record_count,
                    created_at=excluded.created_at,
                    error=excluded.error,
                    deleted_at=excluded.deleted_at,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    str(dataset.get("dataset_id", "")),
                    str(dataset.get("name", "")),
                    str(dataset.get("type", "")),
                    str(dataset.get("status", "")),
                    int(dataset.get("record_count", 0) or 0),
                    int(dataset.get("created_at", now) or now),
                    int(dataset.get("updated_at", now) or now),
                    dataset.get("error"),
                    dataset.get("deleted_at"),
                    json.dumps(dataset, separators=(",", ":"), ensure_ascii=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _sqlite_upsert_eval_run_record(run: Dict[str, Any]) -> None:
    now = _now_ts()
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            conn.execute(
                """
                INSERT INTO eval_runs(run_id, model, latency_ms, created_at, updated_at, deleted_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    model=excluded.model,
                    latency_ms=excluded.latency_ms,
                    created_at=excluded.created_at,
                    deleted_at=excluded.deleted_at,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    str(run.get("run_id", "")),
                    str(run.get("model", "")),
                    int(run.get("latency_ms", 0) or 0),
                    int(run.get("created_at", now) or now),
                    now,
                    run.get("deleted_at"),
                    json.dumps(run, separators=(",", ":"), ensure_ascii=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _sqlite_upsert_batch_eval_run_record(run: Dict[str, Any]) -> None:
    now = _now_ts()
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            conn.execute(
                """
                INSERT INTO batch_eval_runs(run_id, batch_eval_id, dataset_id, status, created_at, started_at, completed_at, updated_at, deleted_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    batch_eval_id=excluded.batch_eval_id,
                    dataset_id=excluded.dataset_id,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    deleted_at=excluded.deleted_at,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    str(run.get("run_id", "")),
                    str(run.get("batch_eval_id", run.get("run_id", ""))),
                    str(run.get("dataset_id", "")),
                    str(run.get("status", "")),
                    int(run.get("created_at", now) or now),
                    run.get("started_at"),
                    run.get("completed_at"),
                    int(run.get("updated_at", now) or now),
                    run.get("deleted_at"),
                    json.dumps(run, separators=(",", ":"), ensure_ascii=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _sqlite_upsert_rag_index_record(index_obj: Dict[str, Any]) -> None:
    now = _now_ts()
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            conn.execute(
                """
                INSERT INTO rag_indexes(index_id, dataset_id, status, chunk_count, updated_at, deleted_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_id) DO UPDATE SET
                    dataset_id=excluded.dataset_id,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    deleted_at=excluded.deleted_at,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    str(index_obj.get("index_id", "")),
                    str(index_obj.get("dataset_id", "")),
                    str(index_obj.get("status", "")),
                    int(index_obj.get("chunk_count", 0) or 0),
                    int(index_obj.get("updated_at", now) or now),
                    index_obj.get("deleted_at"),
                    json.dumps(index_obj, separators=(",", ":"), ensure_ascii=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _sqlite_insert_audit_event(event: Dict[str, Any]) -> None:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            now = _now_ts()
            conn.execute(
                """
                INSERT INTO audit_logs(event_id, ts, actor_role, actor_key, action, resource_type, resource_id, request_id, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    ts=excluded.ts,
                    actor_role=excluded.actor_role,
                    actor_key=excluded.actor_key,
                    action=excluded.action,
                    resource_type=excluded.resource_type,
                    resource_id=excluded.resource_id,
                    request_id=excluded.request_id,
                    payload=excluded.payload
                """,
                (
                    str(event.get("event_id", "")),
                    int(event.get("ts", now) or now),
                    event.get("actor_role"),
                    event.get("actor_key"),
                    str(event.get("action", "")),
                    str(event.get("resource_type", "")),
                    event.get("resource_id"),
                    event.get("request_id"),
                    json.dumps(event, separators=(",", ":"), ensure_ascii=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _sqlite_query_audit_logs(
    limit: int,
    action: str | None = None,
    resource_type: str | None = None,
    since_ts: int | None = None,
) -> Optional[List[Dict[str, Any]]]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return None
    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_sqlite_schema(conn)
            clauses: List[str] = []
            params: List[Any] = []
            if action:
                clauses.append("action = ?")
                params.append(action)
            if resource_type:
                clauses.append("resource_type = ?")
                params.append(resource_type)
            if since_ts is not None:
                clauses.append("ts >= ?")
                params.append(int(since_ts))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = f"SELECT payload FROM audit_logs {where} ORDER BY ts DESC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for row in rows:
            raw = row["payload"]
            if isinstance(raw, str) and raw:
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        out.append(loaded)
                except Exception:
                    continue
        return out
    except Exception as exc:
        log_json(LOGGER, {"event": "sqlite_audit_query_failed", "error": repr(exc)})
        return None


def _sqlite_soft_delete_by_pk(table: str, pk_column: str, pk_value: str, deleted_at: int) -> int:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            cur = conn.execute(
                f"UPDATE {table} SET deleted_at = ?, updated_at = ? WHERE {pk_column} = ? AND deleted_at IS NULL",
                (deleted_at, deleted_at, pk_value),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()


def _sqlite_soft_delete_dataset_and_related(dataset_id: str, deleted_at: int) -> int:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            cur_ds = conn.execute(
                "UPDATE datasets SET deleted_at = ?, updated_at = ? WHERE dataset_id = ? AND deleted_at IS NULL",
                (deleted_at, deleted_at, dataset_id),
            )
            conn.execute(
                """
                UPDATE rag_indexes
                SET deleted_at = ?, updated_at = ?
                WHERE (index_id = ? OR dataset_id = ?) AND deleted_at IS NULL
                """,
                (deleted_at, deleted_at, dataset_id, dataset_id),
            )
            conn.commit()
            return int(cur_ds.rowcount or 0)
        finally:
            conn.close()


def _sqlite_hard_delete_by_pk(table: str, pk_column: str, pk_value: str) -> int:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            cur = conn.execute(f"DELETE FROM {table} WHERE {pk_column} = ?", (pk_value,))
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()


def _sqlite_hard_delete_dataset_and_related(dataset_id: str) -> int:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            cur_ds = conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            conn.execute("DELETE FROM rag_indexes WHERE index_id = ? OR dataset_id = ?", (dataset_id, dataset_id))
            conn.commit()
            return int(cur_ds.rowcount or 0)
        finally:
            conn.close()


def _sqlite_restore_by_pk(table: str, pk_column: str, pk_value: str, restored_at: int) -> int:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            cur = conn.execute(
                f"UPDATE {table} SET deleted_at = NULL, updated_at = ? WHERE {pk_column} = ? AND deleted_at IS NOT NULL",
                (restored_at, pk_value),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()


def _sqlite_restore_dataset_and_related(dataset_id: str, restored_at: int) -> int:
    with STATE_LOCK:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            cur_ds = conn.execute(
                "UPDATE datasets SET deleted_at = NULL, updated_at = ? WHERE dataset_id = ? AND deleted_at IS NOT NULL",
                (restored_at, dataset_id),
            )
            conn.execute(
                """
                UPDATE rag_indexes
                SET deleted_at = NULL, updated_at = ?
                WHERE (index_id = ? OR dataset_id = ?) AND deleted_at IS NOT NULL
                """,
                (restored_at, dataset_id, dataset_id),
            )
            conn.commit()
            return int(cur_ds.rowcount or 0)
        finally:
            conn.close()


def _persist_dataset_record(dataset: Dict[str, Any]) -> None:
    if not STATE_PERSISTENCE_ENABLED:
        return
    if STATE_BACKEND == "sqlite":
        try:
            _sqlite_upsert_dataset_record(dataset)
            return
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_dataset_persist_failed", "error": repr(exc)})
    _save_state_to_disk()


def _persist_eval_run_record(run: Dict[str, Any]) -> None:
    if not STATE_PERSISTENCE_ENABLED:
        return
    if STATE_BACKEND == "sqlite":
        try:
            _sqlite_upsert_eval_run_record(run)
            return
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_eval_persist_failed", "error": repr(exc)})
    _save_state_to_disk()


def _persist_batch_eval_run_record(run: Dict[str, Any]) -> None:
    if not STATE_PERSISTENCE_ENABLED:
        return
    if STATE_BACKEND == "sqlite":
        try:
            _sqlite_upsert_batch_eval_run_record(run)
            return
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_batch_persist_failed", "error": repr(exc)})
    _save_state_to_disk()


def _persist_rag_index_record(index_obj: Dict[str, Any]) -> None:
    if not STATE_PERSISTENCE_ENABLED:
        return
    if STATE_BACKEND == "sqlite":
        try:
            _sqlite_upsert_rag_index_record(index_obj)
            return
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_rag_index_persist_failed", "error": repr(exc)})
    _save_state_to_disk()


def _record_audit_event(
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    auth = AUTH_CONTEXT.get() or {}
    event = {
        "event_id": f"aud_{uuid.uuid4().hex[:12]}",
        "ts": _now_ts(),
        "request_id": REQUEST_ID.get(),
        "actor_role": auth.get("role", "unknown"),
        "actor_key": _audit_key_fingerprint(auth.get("api_key")),
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details or {},
    }
    AUDIT_LOGS.append(event)
    _trim_audit_logs()

    if STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite":
        try:
            _sqlite_insert_audit_event(event)
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_audit_persist_failed", "error": repr(exc)})
            _save_state_to_disk()
    elif STATE_PERSISTENCE_ENABLED:
        _save_state_to_disk()

    return event


def _query_audit_logs(
    limit: int = 100,
    action: str | None = None,
    resource_type: str | None = None,
    since_ts: int | None = None,
) -> List[Dict[str, Any]]:
    clamped_limit = max(1, min(int(limit), 500))
    sqlite_rows = _sqlite_query_audit_logs(
        limit=clamped_limit,
        action=action,
        resource_type=resource_type,
        since_ts=since_ts,
    )
    if sqlite_rows is not None:
        return sqlite_rows

    rows = list(AUDIT_LOGS)
    if action:
        rows = [r for r in rows if r.get("action") == action]
    if resource_type:
        rows = [r for r in rows if r.get("resource_type") == resource_type]
    if since_ts is not None:
        rows = [r for r in rows if int(r.get("ts", 0) or 0) >= int(since_ts)]
    rows = sorted(rows, key=lambda x: int(x.get("ts", 0) or 0), reverse=True)
    return rows[:clamped_limit]


def _delete_dataset_record(dataset_id: str) -> bool:
    deleted_at = _now_ts()
    existed = False
    ds = DATASETS.get(dataset_id)
    if ds is not None and ds.get("deleted_at") is None:
        ds["deleted_at"] = deleted_at
        ds["updated_at"] = deleted_at
        existed = True
    idx = RAG_INDEXES.get(dataset_id)
    if idx is not None and idx.get("deleted_at") is None:
        idx["deleted_at"] = deleted_at
        idx["updated_at"] = deleted_at
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_soft_delete_dataset_and_related(dataset_id, deleted_at) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_dataset_delete_failed", "error": repr(exc), "dataset_id": dataset_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _delete_eval_run_record(run_id: str) -> bool:
    deleted_at = _now_ts()
    existed = False
    run = EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is None:
        run["deleted_at"] = deleted_at
        existed = True
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_soft_delete_by_pk("eval_runs", "run_id", run_id, deleted_at) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_eval_delete_failed", "error": repr(exc), "run_id": run_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _delete_batch_eval_run_record(run_id: str) -> bool:
    deleted_at = _now_ts()
    existed = False
    run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is None:
        run["deleted_at"] = deleted_at
        run["updated_at"] = deleted_at
        _append_batch_event(run, event_type="deleted", details={"deleted_at": deleted_at})
        _persist_batch_eval_run_record(run)
        existed = True
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_soft_delete_by_pk("batch_eval_runs", "run_id", run_id, deleted_at) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_batch_delete_failed", "error": repr(exc), "run_id": run_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _delete_rag_index_record(index_id: str) -> bool:
    deleted_at = _now_ts()
    existed = False
    idx = RAG_INDEXES.get(index_id)
    if idx is not None and idx.get("deleted_at") is None:
        idx["deleted_at"] = deleted_at
        idx["updated_at"] = deleted_at
        existed = True
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_soft_delete_by_pk("rag_indexes", "index_id", index_id, deleted_at) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_rag_index_delete_failed", "error": repr(exc), "index_id": index_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _restore_dataset_record(dataset_id: str) -> bool:
    restored_at = _now_ts()
    restored = False
    ds = DATASETS.get(dataset_id)
    if ds is not None and ds.get("deleted_at") is not None:
        ds["deleted_at"] = None
        ds["updated_at"] = restored_at
        restored = True
    idx = RAG_INDEXES.get(dataset_id)
    if idx is not None and idx.get("deleted_at") is not None:
        idx["deleted_at"] = None
        idx["updated_at"] = restored_at

    if not STATE_PERSISTENCE_ENABLED:
        return restored
    if STATE_BACKEND == "sqlite":
        try:
            restored_sql = _sqlite_restore_dataset_and_related(dataset_id, restored_at) > 0
            return restored or restored_sql
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_dataset_restore_failed", "error": repr(exc), "dataset_id": dataset_id})
            _save_state_to_disk()
            return restored
    _save_state_to_disk()
    return restored


def _restore_eval_run_record(run_id: str) -> bool:
    restored_at = _now_ts()
    restored = False
    run = EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None:
        run["deleted_at"] = None
        restored = True
    if not STATE_PERSISTENCE_ENABLED:
        return restored
    if STATE_BACKEND == "sqlite":
        try:
            restored_sql = _sqlite_restore_by_pk("eval_runs", "run_id", run_id, restored_at) > 0
            return restored or restored_sql
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_eval_restore_failed", "error": repr(exc), "run_id": run_id})
            _save_state_to_disk()
            return restored
    _save_state_to_disk()
    return restored


def _restore_batch_eval_run_record(run_id: str) -> bool:
    restored_at = _now_ts()
    restored = False
    run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None:
        run["deleted_at"] = None
        run["updated_at"] = restored_at
        _append_batch_event(run, event_type="restored", details={"restored_at": restored_at})
        _persist_batch_eval_run_record(run)
        restored = True
    if not STATE_PERSISTENCE_ENABLED:
        return restored
    if STATE_BACKEND == "sqlite":
        try:
            restored_sql = _sqlite_restore_by_pk("batch_eval_runs", "run_id", run_id, restored_at) > 0
            return restored or restored_sql
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_batch_restore_failed", "error": repr(exc), "run_id": run_id})
            _save_state_to_disk()
            return restored
    _save_state_to_disk()
    return restored


def _restore_rag_index_record(index_id: str) -> bool:
    restored_at = _now_ts()
    restored = False
    idx = RAG_INDEXES.get(index_id)
    if idx is not None and idx.get("deleted_at") is not None:
        idx["deleted_at"] = None
        idx["updated_at"] = restored_at
        restored = True
    if not STATE_PERSISTENCE_ENABLED:
        return restored
    if STATE_BACKEND == "sqlite":
        try:
            restored_sql = _sqlite_restore_by_pk("rag_indexes", "index_id", index_id, restored_at) > 0
            return restored or restored_sql
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_rag_index_restore_failed", "error": repr(exc), "index_id": index_id})
            _save_state_to_disk()
            return restored
    _save_state_to_disk()
    return restored


def _purge_dataset_record(dataset_id: str) -> bool:
    existed = DATASETS.pop(dataset_id, None) is not None
    RAG_INDEXES.pop(dataset_id, None)
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_hard_delete_dataset_and_related(dataset_id) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_dataset_purge_failed", "error": repr(exc), "dataset_id": dataset_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _purge_eval_run_record(run_id: str) -> bool:
    existed = EVAL_RUNS.pop(run_id, None) is not None
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_hard_delete_by_pk("eval_runs", "run_id", run_id) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_eval_purge_failed", "error": repr(exc), "run_id": run_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _purge_batch_eval_run_record(run_id: str) -> bool:
    existed = BATCH_EVAL_RUNS.pop(run_id, None) is not None
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_hard_delete_by_pk("batch_eval_runs", "run_id", run_id) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_batch_purge_failed", "error": repr(exc), "run_id": run_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _purge_rag_index_record(index_id: str) -> bool:
    existed = RAG_INDEXES.pop(index_id, None) is not None
    if not STATE_PERSISTENCE_ENABLED:
        return existed
    if STATE_BACKEND == "sqlite":
        try:
            deleted = _sqlite_hard_delete_by_pk("rag_indexes", "index_id", index_id) > 0
            return existed or deleted
        except Exception as exc:
            log_json(LOGGER, {"event": "sqlite_rag_index_purge_failed", "error": repr(exc), "index_id": index_id})
            _save_state_to_disk()
            return existed
    _save_state_to_disk()
    return existed


def _sqlite_expired_soft_deleted_ids(table: str, pk_column: str, cutoff_ts: int) -> List[str]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return []
    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            rows = conn.execute(
                f"SELECT {pk_column} FROM {table} WHERE deleted_at IS NOT NULL AND deleted_at <= ?",
                (cutoff_ts,),
            ).fetchall()
        finally:
            conn.close()
        return [str(row[0]) for row in rows]
    except Exception as exc:
        log_json(LOGGER, {"event": "sqlite_retention_scan_failed", "table": table, "error": repr(exc)})
        return []


def _collect_expired_soft_deleted_ids(cutoff_ts: int) -> Dict[str, set[str]]:
    def _is_expired_deleted(value: Any) -> bool:
        try:
            iv = int(value)
        except Exception:
            return False
        return iv <= cutoff_ts

    dataset_ids = {k for k, v in DATASETS.items() if _is_expired_deleted(v.get("deleted_at"))}
    eval_ids = {k for k, v in EVAL_RUNS.items() if _is_expired_deleted(v.get("deleted_at"))}
    batch_ids = {k for k, v in BATCH_EVAL_RUNS.items() if _is_expired_deleted(v.get("deleted_at"))}
    rag_ids = {k for k, v in RAG_INDEXES.items() if _is_expired_deleted(v.get("deleted_at"))}

    dataset_ids.update(_sqlite_expired_soft_deleted_ids("datasets", "dataset_id", cutoff_ts))
    eval_ids.update(_sqlite_expired_soft_deleted_ids("eval_runs", "run_id", cutoff_ts))
    batch_ids.update(_sqlite_expired_soft_deleted_ids("batch_eval_runs", "run_id", cutoff_ts))
    rag_ids.update(_sqlite_expired_soft_deleted_ids("rag_indexes", "index_id", cutoff_ts))

    return {
        "dataset_ids": dataset_ids,
        "eval_run_ids": eval_ids,
        "batch_eval_run_ids": batch_ids,
        "rag_index_ids": rag_ids,
    }


def _preview_expired_soft_deleted_records(
    retention_seconds: Optional[int] = None,
    now_ts: Optional[int] = None,
    include_ids: bool = True,
    candidate_ids_limit: int = 100,
) -> Dict[str, Any]:
    ttl = SOFT_DELETE_RETENTION_SECONDS if retention_seconds is None else int(retention_seconds)
    now = _now_ts() if now_ts is None else int(now_ts)
    cutoff = now - max(ttl, 0)
    candidates = _collect_expired_soft_deleted_ids(cutoff)
    max_ids = max(0, int(candidate_ids_limit))

    ds_ids = sorted(candidates["dataset_ids"])
    ev_ids = sorted(candidates["eval_run_ids"])
    be_ids = sorted(candidates["batch_eval_run_ids"])
    rg_ids = sorted(candidates["rag_index_ids"])

    ds_view = ds_ids[:max_ids] if include_ids else []
    ev_view = ev_ids[:max_ids] if include_ids else []
    be_view = be_ids[:max_ids] if include_ids else []
    rg_view = rg_ids[:max_ids] if include_ids else []

    return {
        "retention_seconds": max(ttl, 0),
        "cutoff_ts": cutoff,
        "candidate_datasets": len(candidates["dataset_ids"]),
        "candidate_eval_runs": len(candidates["eval_run_ids"]),
        "candidate_batch_eval_runs": len(candidates["batch_eval_run_ids"]),
        "candidate_rag_indexes": len(candidates["rag_index_ids"]),
        "candidate_total": (
            len(candidates["dataset_ids"])
            + len(candidates["eval_run_ids"])
            + len(candidates["batch_eval_run_ids"])
            + len(candidates["rag_index_ids"])
        ),
        "include_ids": include_ids,
        "candidate_ids_limit": max_ids,
        "candidate_ids_truncated": (
            include_ids
            and (
                len(ds_ids) > len(ds_view)
                or len(ev_ids) > len(ev_view)
                or len(be_ids) > len(be_view)
                or len(rg_ids) > len(rg_view)
            )
        ),
        "candidates": {
            "dataset_ids": ds_view,
            "eval_run_ids": ev_view,
            "batch_eval_run_ids": be_view,
            "rag_index_ids": rg_view,
        },
    }


def _purge_expired_soft_deleted_records(
    retention_seconds: Optional[int] = None,
    now_ts: Optional[int] = None,
) -> Dict[str, int]:
    ttl = SOFT_DELETE_RETENTION_SECONDS if retention_seconds is None else int(retention_seconds)
    now = _now_ts() if now_ts is None else int(now_ts)
    cutoff = now - max(ttl, 0)
    candidates = _collect_expired_soft_deleted_ids(cutoff)
    dataset_ids = candidates["dataset_ids"]
    eval_ids = candidates["eval_run_ids"]
    batch_ids = candidates["batch_eval_run_ids"]
    rag_ids = candidates["rag_index_ids"]

    purged_datasets = 0
    purged_evals = 0
    purged_batches = 0
    purged_rag_indexes = 0

    for dataset_id in dataset_ids:
        if _purge_dataset_record(dataset_id):
            purged_datasets += 1
    for run_id in eval_ids:
        if _purge_eval_run_record(run_id):
            purged_evals += 1
    for run_id in batch_ids:
        if _purge_batch_eval_run_record(run_id):
            purged_batches += 1
    for index_id in rag_ids:
        if _purge_rag_index_record(index_id):
            purged_rag_indexes += 1

    return {
        "retention_seconds": max(ttl, 0),
        "cutoff_ts": cutoff,
        "purged_datasets": purged_datasets,
        "purged_eval_runs": purged_evals,
        "purged_batch_eval_runs": purged_batches,
        "purged_rag_indexes": purged_rag_indexes,
        "purged_total": purged_datasets + purged_evals + purged_batches + purged_rag_indexes,
    }


def _record_retention_run(result: Dict[str, Any], error: Optional[str], trigger: str) -> None:
    ts = _now_ts()
    purged_total = int(result.get("purged_total", 0) or 0)
    RETENTION_STATS["last_run_ts"] = ts
    RETENTION_STATS["last_purged_total"] = purged_total
    RETENTION_STATS["last_error"] = error

    RETENTION_HISTORY.append(
        {
            "ts": ts,
            "trigger": trigger,
            "purged_total": purged_total,
            "retention_seconds": int(result.get("retention_seconds", max(SOFT_DELETE_RETENTION_SECONDS, 0)) or 0),
            "cutoff_ts": int(result.get("cutoff_ts", ts) or ts),
            "error": error,
        }
    )
    max_len = max(1, RETENTION_HISTORY_LIMIT)
    if len(RETENTION_HISTORY) > max_len:
        del RETENTION_HISTORY[:-max_len]


def _retention_sweep_loop() -> None:
    interval = max(RETENTION_SWEEP_INTERVAL_SECONDS, 5)
    while not RETENTION_STOP_EVENT.wait(interval):
        try:
            result = _purge_expired_soft_deleted_records()
            _record_retention_run(result=result, error=None, trigger="background")
            if result["purged_total"] > 0:
                log_json(LOGGER, {"event": "retention_sweep_purged", **result})
        except Exception as exc:
            _record_retention_run(
                result={"retention_seconds": max(SOFT_DELETE_RETENTION_SECONDS, 0), "cutoff_ts": _now_ts(), "purged_total": 0},
                error=repr(exc),
                trigger="background",
            )
            log_json(LOGGER, {"event": "retention_sweep_failed", "error": repr(exc)})


def _save_state_to_disk() -> None:
    if not STATE_PERSISTENCE_ENABLED:
        return

    try:
        if STATE_BACKEND == "sqlite":
            _save_state_sqlite()
        else:
            _save_state_json()
    except Exception as exc:
        log_json(
            LOGGER,
            {
                "event": "state_persist_failed",
                "error": repr(exc),
                "state_backend": _state_backend_name(),
            },
        )


def _load_state_from_disk() -> None:
    if not STATE_PERSISTENCE_ENABLED:
        return

    try:
        if STATE_BACKEND == "sqlite":
            _load_state_sqlite()
        else:
            _load_state_json()

        log_json(
            LOGGER,
            {
                "event": "state_loaded",
                "state_backend": _state_backend_name(),
                "datasets": len(DATASETS),
                "eval_runs": len(EVAL_RUNS),
                "batch_eval_runs": len(BATCH_EVAL_RUNS),
            },
        )
    except Exception as exc:
        log_json(
            LOGGER,
            {
                "event": "state_load_failed",
                "error": repr(exc),
                "state_backend": _state_backend_name(),
            },
        )


def _sqlite_dataset_rows(
    dataset_type: str | None,
    status: str | None,
    limit: int,
    include_deleted: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return None

    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_sqlite_schema(conn)
            clauses = []
            params: List[Any] = []
            if dataset_type is not None:
                clauses.append("type = ?")
                params.append(dataset_type)
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            if not include_deleted:
                clauses.append("deleted_at IS NULL")
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = (
                "SELECT dataset_id, name, type, status, record_count, created_at, updated_at, error, deleted_at, payload "
                f"FROM datasets {where} ORDER BY created_at DESC LIMIT ?"
            )
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        out: List[Dict[str, Any]] = []
        for row in rows:
            row_payload = {}
            raw_payload = row["payload"]
            if isinstance(raw_payload, str) and raw_payload:
                try:
                    row_payload = json.loads(raw_payload)
                except Exception:
                    row_payload = {}
            out.append(
                {
                    "dataset_id": row["dataset_id"] or row_payload.get("dataset_id"),
                    "name": row["name"] or row_payload.get("name"),
                    "type": row["type"] or row_payload.get("type"),
                    "status": row["status"] or row_payload.get("status"),
                    "record_count": row["record_count"] if row["record_count"] is not None else row_payload.get("record_count", 0),
                    "created_at": row["created_at"] if row["created_at"] is not None else row_payload.get("created_at", 0),
                    "updated_at": row["updated_at"] if row["updated_at"] is not None else row_payload.get("updated_at", 0),
                    "error": row["error"] if row["error"] is not None else row_payload.get("error"),
                    "deleted_at": row["deleted_at"] if row["deleted_at"] is not None else row_payload.get("deleted_at"),
                    "records": row_payload.get("records", []),
                    "metadata": row_payload.get("metadata", {}),
                }
            )
        return out
    except Exception as exc:
        log_json(LOGGER, {"event": "sqlite_dataset_query_failed", "error": repr(exc)})
        return None


def _sqlite_dataset_row(dataset_id: str, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return None
    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_sqlite_schema(conn)
            row = conn.execute(
                """
                SELECT dataset_id, name, type, status, record_count, created_at, updated_at, error, deleted_at, payload
                FROM datasets
                WHERE dataset_id = ?
                LIMIT 1
                """,
                (dataset_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        if (not include_deleted) and row["deleted_at"] is not None:
            return None

        row_payload = {}
        raw_payload = row["payload"]
        if isinstance(raw_payload, str) and raw_payload:
            try:
                row_payload = json.loads(raw_payload)
            except Exception:
                row_payload = {}

        return {
            "dataset_id": row["dataset_id"] or row_payload.get("dataset_id"),
            "name": row["name"] or row_payload.get("name"),
            "type": row["type"] or row_payload.get("type"),
            "status": row["status"] or row_payload.get("status"),
            "record_count": row["record_count"] if row["record_count"] is not None else row_payload.get("record_count", 0),
            "created_at": row["created_at"] if row["created_at"] is not None else row_payload.get("created_at", 0),
            "updated_at": row["updated_at"] if row["updated_at"] is not None else row_payload.get("updated_at", 0),
            "error": row["error"] if row["error"] is not None else row_payload.get("error"),
            "deleted_at": row["deleted_at"] if row["deleted_at"] is not None else row_payload.get("deleted_at"),
            "records": row_payload.get("records", []),
            "metadata": row_payload.get("metadata", {}),
        }
    except Exception as exc:
        log_json(LOGGER, {"event": "sqlite_dataset_lookup_failed", "error": repr(exc), "dataset_id": dataset_id})
        return None


def _sqlite_payload_by_pk(
    table: str,
    pk_column: str,
    pk_value: str,
    event_name: str,
    include_deleted: bool = False,
) -> Optional[Dict[str, Any]]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return None

    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_sqlite_schema(conn)
            row = conn.execute(
                f"SELECT payload, deleted_at FROM {table} WHERE {pk_column} = ? LIMIT 1",
                (pk_value,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        if (not include_deleted) and row["deleted_at"] is not None:
            return None

        payload = row["payload"]
        if not isinstance(payload, str) or not payload:
            return None
        loaded = json.loads(payload)
        if isinstance(loaded, dict):
            return loaded
        return None
    except Exception as exc:
        log_json(LOGGER, {"event": event_name, "error": repr(exc), "id": pk_value})
        return None


def _sqlite_eval_run(run_id: str, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    return _sqlite_payload_by_pk(
        "eval_runs", "run_id", run_id, "sqlite_eval_lookup_failed", include_deleted=include_deleted
    )


def _sqlite_batch_eval_run(run_id: str, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    return _sqlite_payload_by_pk(
        "batch_eval_runs", "run_id", run_id, "sqlite_batch_lookup_failed", include_deleted=include_deleted
    )


def _sqlite_rag_index(index_id: str, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    return _sqlite_payload_by_pk(
        "rag_indexes", "index_id", index_id, "sqlite_rag_lookup_failed", include_deleted=include_deleted
    )


def _sqlite_state_counts() -> Optional[Dict[str, int]]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return None
    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            datasets_count = int(conn.execute("SELECT COUNT(*) FROM datasets WHERE deleted_at IS NULL").fetchone()[0])
            eval_runs_count = int(conn.execute("SELECT COUNT(*) FROM eval_runs WHERE deleted_at IS NULL").fetchone()[0])
            batch_runs_count = int(
                conn.execute("SELECT COUNT(*) FROM batch_eval_runs WHERE deleted_at IS NULL").fetchone()[0]
            )
            rag_indexes_count = int(
                conn.execute("SELECT COUNT(*) FROM rag_indexes WHERE deleted_at IS NULL").fetchone()[0]
            )
            audit_logs_count = int(conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0])
            runbook_runs_count = int(conn.execute("SELECT COUNT(*) FROM runbook_runs").fetchone()[0])
            runbook_templates_count = int(conn.execute("SELECT COUNT(*) FROM runbook_templates").fetchone()[0])
        finally:
            conn.close()
        return {
            "datasets": datasets_count,
            "eval_runs": eval_runs_count,
            "batch_eval_runs": batch_runs_count,
            "rag_indexes": rag_indexes_count,
            "audit_logs": audit_logs_count,
            "runbook_runs": runbook_runs_count,
            "runbook_templates": runbook_templates_count,
        }
    except Exception as exc:
        log_json(LOGGER, {"event": "sqlite_counts_failed", "error": repr(exc)})
        return None


def _sqlite_batch_queue_depth() -> Optional[Dict[str, int]]:
    if not (STATE_PERSISTENCE_ENABLED and STATE_BACKEND == "sqlite" and os.path.exists(STATE_SQLITE_PATH)):
        return None
    try:
        conn = sqlite3.connect(STATE_SQLITE_PATH)
        try:
            _ensure_sqlite_schema(conn)
            queued = int(
                conn.execute(
                    "SELECT COUNT(*) FROM batch_eval_runs WHERE status = 'queued' AND deleted_at IS NULL"
                ).fetchone()[0]
            )
            running = int(
                conn.execute(
                    "SELECT COUNT(*) FROM batch_eval_runs WHERE status = 'running' AND deleted_at IS NULL"
                ).fetchone()[0]
            )
        finally:
            conn.close()
        return {"pending": queued, "running": running}
    except Exception as exc:
        log_json(LOGGER, {"event": "sqlite_queue_depth_failed", "error": repr(exc)})
        return None


def _state_counts() -> Dict[str, int]:
    sqlite_counts = _sqlite_state_counts()
    if sqlite_counts is not None:
        return sqlite_counts
    return {
        "datasets": sum(1 for v in DATASETS.values() if v.get("deleted_at") is None),
        "eval_runs": sum(1 for v in EVAL_RUNS.values() if v.get("deleted_at") is None),
        "batch_eval_runs": sum(1 for v in BATCH_EVAL_RUNS.values() if v.get("deleted_at") is None),
        "rag_indexes": sum(1 for v in RAG_INDEXES.values() if v.get("deleted_at") is None),
        "audit_logs": len(AUDIT_LOGS),
        "runbook_runs": len(RUNBOOK_RUNS),
        "runbook_templates": len(RUNBOOK_TEMPLATES),
    }


def _batch_queue_depth() -> Dict[str, int]:
    sqlite_depth = _sqlite_batch_queue_depth()
    if sqlite_depth is not None:
        return sqlite_depth
    pending = 0
    running = 0
    for run in BATCH_EVAL_RUNS.values():
        if run.get("deleted_at") is not None:
            continue
        status = run.get("status")
        if status == "queued":
            pending += 1
        elif status == "running":
            running += 1
    return {"pending": pending, "running": running}


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _get_model_type() -> str:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "unknown"
    if getattr(cfg, "is_encoder_decoder", False):
        return "encoder-decoder"
    return "decoder-only"


def _get_attention_masking() -> str:
    mt = _get_model_type()
    if mt == "decoder-only":
        return "causal"
    if mt == "encoder-decoder":
        return "unknown"
    return "unknown"


def _get_context_window() -> int:
    cfg = getattr(model, "config", None)
    candidates = []

    if cfg is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            iv = _safe_int(getattr(cfg, attr, None))
            if iv:
                candidates.append(iv)

    tok_max = _safe_int(getattr(tokenizer, "model_max_length", None))
    if tok_max and tok_max < 100_000:
        candidates.append(tok_max)

    candidates = [c for c in candidates if c >= 128]
    if not candidates:
        return 2048
    return min(candidates)


def _count_tokens(text: str) -> int:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return int(ids.numel())


def _get_heads_and_hidden() -> Tuple[Optional[int], Optional[int]]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None, None

    heads = getattr(cfg, "num_attention_heads", None)
    if heads is None:
        heads = getattr(cfg, "n_head", None)

    hidden = getattr(cfg, "hidden_size", None)
    if hidden is None:
        hidden = getattr(cfg, "n_embd", None)
    if hidden is None:
        hidden = getattr(cfg, "d_model", None)

    return _safe_int(heads), _safe_int(hidden)


def _estimated_attention_ops(seq_len: int) -> int:
    return int(seq_len) * int(seq_len)


def _enforce_context_guardrail(prompt_tokens: int, max_new_tokens: int, context_window: int) -> None:
    total = prompt_tokens + max_new_tokens
    if total > context_window:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "prompt_plus_generation_exceeds_context_window",
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "context_window": context_window,
                "total_tokens_requested": total,
                "hint": "Reduce prompt length or max_new_tokens.",
            },
        )


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Tuple[str, int, int]:
    if not TORCH_AVAILABLE:
        raise HTTPException(status_code=503, detail={"code": "service_unavailable", "message": "torch is not available"})

    prompt_tokens = _count_tokens(prompt)

    context_window = MODEL_META["context_window"]
    _enforce_context_guardrail(prompt_tokens, max_new_tokens, context_window)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    do_sample = temperature > 0.0

    try:
        inference_ctx = torch.inference_mode() if TORCH_AVAILABLE else nullcontext()
        with inference_ctx:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as e:
        # Make this a clean 400 so stress tests don't look like "server crashed"
        raise HTTPException(status_code=400, detail={"error": "generation_failed", "detail": repr(e)})

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])

    if full_text.startswith(prompt):
        response_text = full_text[len(prompt):].lstrip()
    else:
        response_text = full_text.strip()

    return response_text, prompt_tokens, completion_tokens


def _build_critique_prompt(user_prompt: str, answer: str) -> str:
    return (
        "<|system|>\n"
        "You are a strict technical reviewer. "
        "Do not restate the answer. "
        "Only list concrete errors or weaknesses.\n"
        "<|user|>\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "List flaws in bullet points.\n"
        "<|assistant|>\n"
    )


def _build_refine_prompt(user_prompt: str, answer: str, critique: str) -> str:
    return (
        "<|system|>\n"
        "You are rewriting the answer. "
        "Fix the flaws listed in the critique. "
        "Do not add conversational phrases. "
        "Do not say 'I hope this helps'. "
        "Be concise and technically accurate.\n"
        "<|user|>\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Original answer:\n{answer}\n\n"
        f"Critique:\n{critique}\n\n"
        "Rewrite the improved answer below:\n"
        "<|assistant|>\n"
    )


def _build_eval_prompt(user_prompt: str, answer: str, criteria: List[str]) -> str:
    crit = ", ".join(criteria)
    return (
        "<|system|>\n"
        "You are an evaluator. Score the answer 1-10 for each criterion. "
        "Return EXACTLY this format with no extra text:\n"
        "criterion: <name>\n"
        "score: <number>\n"
        "rationale: <one sentence>\n"
        "---\n"
        "Repeat block for each criterion.\n"
        "<|user|>\n"
        f"Criteria: {crit}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Assistant response:\n{answer}\n"
        "<|assistant|>\n"
    )


def _extract_json_block(text: str) -> Optional[str]:
    """
    Best-effort extraction of the first JSON object.
    Keeps this stage lightweight. We can harden later.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _cache_key(payload: ChatRequest) -> str:
    return "|".join(
        [
            payload.prompt,
            str(payload.max_new_tokens),
            str(payload.temperature),
            str(payload.top_p),
            payload.mode,
            str(payload.refine_steps),
            str(payload.critique_temperature),
            MODEL_NAME,
        ]
    )


def _redis_cache_key(cache_key: str) -> str:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return f"{REDIS_KEY_PREFIX}{digest}"


def _get_cached_chat_response(cache_key: str) -> Optional[ChatResponse]:
    if _cache_backend_name() == "redis" and REDIS_CLIENT is not None:
        try:
            raw = REDIS_CLIENT.get(_redis_cache_key(cache_key))
            if raw is None:
                return None
            loaded = json.loads(raw) if isinstance(raw, str) else raw
            response = ChatResponse.model_validate(loaded)
            response.cache_hit = True
            response.request_id = REQUEST_ID.get()
            return response
        except Exception as exc:
            log_json(LOGGER, {"event": "cache_get_failed", "backend": "redis", "error": repr(exc)})

    cached = CHAT_CACHE.get(cache_key)
    if cached is None:
        return None

    age = time.time() - cached["created_at"]
    if age > CACHE_TTL_SECONDS:
        CHAT_CACHE.pop(cache_key, None)
        return None

    response = ChatResponse.model_validate(cached["response"])
    response.cache_hit = True
    response.request_id = REQUEST_ID.get()
    return response


def _put_cached_chat_response(cache_key: str, response: ChatResponse) -> None:
    if _cache_backend_name() == "redis" and REDIS_CLIENT is not None:
        try:
            REDIS_CLIENT.setex(
                _redis_cache_key(cache_key),
                CACHE_TTL_SECONDS,
                json.dumps(response.model_dump(), separators=(",", ":"), ensure_ascii=True),
            )
            return
        except Exception as exc:
            log_json(LOGGER, {"event": "cache_put_failed", "backend": "redis", "error": repr(exc)})

    CHAT_CACHE[cache_key] = {
        "created_at": time.time(),
        "response": response.model_dump(),
    }


def _cache_stats() -> Dict[str, Any]:
    entries = 0

    if _cache_backend_name() == "redis" and REDIS_CLIENT is not None:
        try:
            for _ in REDIS_CLIENT.scan_iter(match=f"{REDIS_KEY_PREFIX}*"):
                entries += 1
        except Exception as exc:
            log_json(LOGGER, {"event": "cache_scan_failed", "backend": "redis", "error": repr(exc)})
            entries = 0
    else:
        now = time.time()
        active_keys = []
        for key, value in CHAT_CACHE.items():
            if now - value["created_at"] <= CACHE_TTL_SECONDS:
                active_keys.append(key)
        for key in set(CHAT_CACHE.keys()) - set(active_keys):
            CHAT_CACHE.pop(key, None)
        entries = len(active_keys)

    hits = METRICS["chat_cache_hits"]
    misses = METRICS["chat_cache_misses"]
    total = hits + misses
    hit_rate = round(hits / total, 4) if total else 0.0

    return {
        "backend": _cache_backend_name(),
        "ttl_seconds": CACHE_TTL_SECONDS,
        "entries": entries,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
    }


def _now_ts() -> int:
    return int(time.time())

def _iso_utc_from_epoch(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_load_state_from_disk()
_init_cache_backend()
_init_auth_registry()
_init_rag_vector_backend()
_init_tracing_backend()


def _parse_bearer_token(authorization: str | None) -> Optional[str]:
    if authorization is None:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix):].strip()
    return token or None


def _resolve_api_key(x_api_key: str | None, authorization: str | None) -> Optional[str]:
    if x_api_key:
        return x_api_key
    return _parse_bearer_token(authorization)


def _parse_dataset_upload(filename: str, raw: bytes) -> List[Dict[str, Any]]:
    lowered = filename.lower()

    if lowered.endswith('.jsonl'):
        rows: List[Dict[str, Any]] = []
        for line in raw.decode('utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
        return rows

    if lowered.endswith('.csv'):
        text = raw.decode('utf-8')
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    if lowered.endswith('.txt'):
        text = raw.decode('utf-8')
        return [{'text': ln.strip()} for ln in text.splitlines() if ln.strip()]

    raise HTTPException(
        status_code=400,
        detail={
            'code': 'invalid_request',
            'message': 'Unsupported file type',
            'details': {'allowed': ['.jsonl', '.csv', '.txt']},
        },
    )


def _validate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(
                {
                    'index': idx,
                    'record_id': None,
                    'code': 'invalid_field_type',
                    'message': 'record must be an object',
                    'path': f'records[{idx}]',
                    'severity': 'error',
                }
            )
            continue

        prompt = ''
        if isinstance(record.get('input'), dict):
            prompt = str(record['input'].get('prompt', '')).strip()
        if not prompt:
            prompt = str(record.get('prompt') or record.get('text') or record.get('input') or '').strip()

        if not prompt:
            errors.append(
                {
                    'index': idx,
                    'record_id': record.get('record_id'),
                    'code': 'missing_required_field',
                    'message': 'record must include prompt/text/input',
                    'path': f'records[{idx}]',
                    'severity': 'error',
                }
            )
            continue

        accepted.append(record)

    return accepted, errors


def _build_batch_progress(total: int, completed: int = 0, failed: int = 0) -> Dict[str, int]:
    return {"total": total, "completed": completed, "failed": failed}


def _append_batch_event(
    run: Dict[str, Any],
    event_type: str,
    message: str | None = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    events = run.get("events")
    if not isinstance(events, list):
        events = []
        run["events"] = events
    events.append(
        {
            "ts": _now_ts(),
            "event_type": event_type,
            "message": message,
            "details": details or {},
        }
    )
    max_len = max(50, BATCH_EVENT_HISTORY_LIMIT)
    if len(events) > max_len:
        del events[:-max_len]


def _finalize_batch_run_as_failed(
    run: Dict[str, Any],
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    now = _now_ts()
    total = int(run.get("progress", {}).get("total", 0) or 0)
    run["status"] = "failed"
    run["progress"] = _build_batch_progress(total=total, completed=int(run.get("progress", {}).get("completed", 0) or 0), failed=total)
    summary = run.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    summary["error"] = message
    summary["total_items"] = total
    summary["failed_items"] = total
    run["summary"] = summary
    run["updated_at"] = now
    run["completed_at"] = now
    _append_batch_event(run, event_type="failed", message=message, details=details or {})


def _slice_batch_events(
    run: Dict[str, Any],
    since_ts: int = 0,
    event_type: str | None = None,
    limit: int = 100,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    events = run.get("events", [])
    if not isinstance(events, list):
        return []

    filtered: List[Dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        ts = int(item.get("ts", 0) or 0)
        if ts <= int(since_ts):
            continue
        if event_type and item.get("event_type") != event_type:
            continue
        filtered.append(item)

    filtered = sorted(filtered, key=lambda x: int(x.get("ts", 0) or 0), reverse=not ascending)
    clamped_limit = max(1, min(int(limit), 1000))
    return filtered[:clamped_limit]


def _to_sse_event(event_name: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _score_record(record: Dict[str, Any], criterion: str) -> int:
    seed = len(str(record)) + len(criterion)
    return (seed % 10) + 1


def _compute_batch_summary(records: List[Dict[str, Any]], criteria: List[str]) -> Dict[str, Any]:
    if not records:
        return {
            "mean_scores": {c: 0.0 for c in criteria},
            "total_items": 0,
            "failed_items": 0,
        }

    sums = {c: 0 for c in criteria}
    failures = 0
    for rec in records:
        failed = False
        for c in criteria:
            s = _score_record(rec, c)
            sums[c] += s
            if c == "overall" and s <= 3:
                failed = True
        if failed:
            failures += 1

    mean_scores = {c: round(sums[c] / len(records), 4) for c in criteria}
    return {
        "mean_scores": mean_scores,
        "total_items": len(records),
        "failed_items": failures,
    }


def _build_batch_record_scores(records: List[Dict[str, Any]], criteria: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        criterion_scores = {c: _score_record(record, c) for c in criteria}
        rows.append({"record_index": idx, "record": record, "scores": criterion_scores})
    return rows


def _compute_distribution(record_scores: List[Dict[str, Any]], criterion: str) -> Dict[str, Any]:
    buckets = {str(i): 0 for i in range(1, 11)}
    values: List[int] = []
    for row in record_scores:
        score = int(row["scores"].get(criterion, 0))
        if 1 <= score <= 10:
            buckets[str(score)] += 1
            values.append(score)

    count = len(values)
    mean_score = round(sum(values) / count, 4) if count else 0.0
    min_score = min(values) if values else 0
    max_score = max(values) if values else 0
    return {
        "buckets": buckets,
        "summary": {
            "count": count,
            "mean": mean_score,
            "min": min_score,
            "max": max_score,
        },
    }


def _score_record_with_retry(
    record: Dict[str, Any],
    criterion: str,
    attempt: int,
    record_index: int,
) -> int:
    # Test hook: simulate transient or persistent scorer failures.
    try:
        transient_failures = int(record.get("_transient_failures", 0) or 0)
    except Exception:
        transient_failures = 0
    raw_always_fail = record.get("_always_fail", False)
    if isinstance(raw_always_fail, bool):
        always_fail = raw_always_fail
    else:
        always_fail = str(raw_always_fail).strip().lower() in {"1", "true", "yes", "on"}
    try:
        sleep_ms = int(record.get("_sleep_ms", 0) or 0)
    except Exception:
        sleep_ms = 0
    if always_fail:
        raise RuntimeError("simulated persistent scorer failure")
    if attempt <= transient_failures:
        raise RuntimeError(f"simulated transient scorer failure on attempt {attempt}")
    if sleep_ms > 0:
        time.sleep(max(0.0, min(float(sleep_ms) / 1000.0, 2.0)))
    return _score_record(record, criterion)


def _dispatch_batch_workers() -> None:
    max_running = max(1, int(BATCH_EVAL_MAX_CONCURRENT_RUNS))
    to_start: List[str] = []
    with BATCH_EVAL_LOCK:
        running = [
            run for run in BATCH_EVAL_RUNS.values()
            if str(run.get("status")) == "running"
        ]
        available = max(0, max_running - len(running))
        if available <= 0:
            return
        queued = sorted(
            [
                run for run in BATCH_EVAL_RUNS.values()
                if str(run.get("status")) == "queued" and not bool(run.get("worker_started", False))
            ],
            key=lambda x: (int(x.get("queue_seq", 0) or 0), int(x.get("created_at", 0) or 0), str(x.get("run_id", ""))),
        )
        for run in queued[:available]:
            run["worker_started"] = True
            run["updated_at"] = _now_ts()
            to_start.append(str(run.get("run_id")))

    for run_id in to_start:
        worker = threading.Thread(target=_run_batch_eval_worker, args=(run_id,), daemon=True, name=f"batch-eval-{run_id}")
        worker.start()


def _get_batch_eval_run_or_404(run_id: str, include_deleted: bool = False) -> Dict[str, Any]:
    run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
    return run


def _batch_failure_criteria_by_record(run: Dict[str, Any]) -> Dict[int, List[str]]:
    failures = run.get("failures", [])
    if not isinstance(failures, list):
        return {}
    out: Dict[int, set[str]] = {}
    for item in failures:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("record_index", -1))
        except Exception:
            continue
        if idx < 0:
            continue
        criterion_name = str(item.get("criterion", "") or "").strip()
        if not criterion_name:
            continue
        bucket = out.get(idx)
        if bucket is None:
            bucket = set()
            out[idx] = bucket
        bucket.add(criterion_name)
    return {k: sorted(v) for k, v in out.items()}


def _build_batch_export_rows(run: Dict[str, Any], include_records: bool = True) -> List[Dict[str, Any]]:
    criteria = [str(c) for c in run.get("criteria", [])]
    record_scores = run.get("record_scores", [])
    if not isinstance(record_scores, list):
        record_scores = []
    failure_map = _batch_failure_criteria_by_record(run)
    rows: List[Dict[str, Any]] = []
    for item in record_scores:
        if not isinstance(item, dict):
            continue
        try:
            record_index = int(item.get("record_index", -1))
        except Exception:
            continue
        scores = item.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}
        failed_criteria = list(failure_map.get(record_index, []))
        row: Dict[str, Any] = {
            "record_index": record_index,
            "failed": bool(failed_criteria),
            "failed_criteria": failed_criteria,
        }
        for criterion_name in criteria:
            raw_score = scores.get(criterion_name, 0)
            try:
                row[f"score_{criterion_name}"] = int(raw_score)
            except Exception:
                row[f"score_{criterion_name}"] = 0
        if include_records:
            row["record"] = item.get("record")
        rows.append(row)
    return rows


def _build_batch_artifact_export_payload(
    run: Dict[str, Any],
    request_id: str,
    include_records: bool,
    failures_limit: int,
) -> Dict[str, Any]:
    clamped_failures_limit = max(1, min(int(failures_limit), 5000))
    failures = run.get("failures", [])
    if not isinstance(failures, list):
        failures = []
    criteria = [str(c) for c in run.get("criteria", [])]
    record_scores = run.get("record_scores", [])
    if not isinstance(record_scores, list):
        record_scores = []
    distribution_by_criterion = {
        criterion_name: _compute_distribution(record_scores, criterion_name) for criterion_name in criteria
    }
    rows = _build_batch_export_rows(run, include_records=include_records)
    retry_stats = run.get("retry_stats", {"attempted_records": 0, "total_retries": 0, "exhausted_records": 0})
    retries = run.get("retries", [])
    if not isinstance(retries, list):
        retries = []
    return {
        "batch_eval_id": run.get("batch_eval_id", run["run_id"]),
        "run_id": run["run_id"],
        "dataset_id": run.get("dataset_id"),
        "status": run.get("status"),
        "criteria": criteria,
        "model": run.get("model"),
        "progress": run.get("progress", {"total": 0, "completed": 0, "failed": 0}),
        "summary": run.get("summary", {"mean_scores": {}, "total_items": 0, "failed_items": 0}),
        "timestamps": {
            "created_at": run.get("created_at"),
            "started_at": run.get("started_at"),
            "updated_at": run.get("updated_at"),
            "completed_at": run.get("completed_at"),
            "deleted_at": run.get("deleted_at"),
        },
        "counts": {
            "record_scores": len(rows),
            "failures_total": len(failures),
            "failures_included": len(failures[:clamped_failures_limit]),
            "retries_total": len(retries),
        },
        "truncated": {
            "failures": len(failures) > clamped_failures_limit,
        },
        "artifacts": {
            "distribution_by_criterion": distribution_by_criterion,
            "failures": list(failures[:clamped_failures_limit]),
            "record_scores": rows,
            "retry_stats": retry_stats,
        },
        "request_id": request_id,
    }


def _render_batch_artifact_export_csv(run: Dict[str, Any], include_records: bool) -> str:
    criteria = [str(c) for c in run.get("criteria", [])]
    rows = _build_batch_export_rows(run, include_records=include_records)
    fieldnames = ["run_id", "dataset_id", "record_index", "failed", "failed_criteria"]
    fieldnames.extend([f"score_{criterion_name}" for criterion_name in criteria])
    if include_records:
        fieldnames.append("record_json")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        out_row: Dict[str, Any] = {
            "run_id": run["run_id"],
            "dataset_id": run.get("dataset_id", ""),
            "record_index": row.get("record_index"),
            "failed": str(bool(row.get("failed"))).lower(),
            "failed_criteria": "|".join(row.get("failed_criteria", [])),
        }
        for criterion_name in criteria:
            out_row[f"score_{criterion_name}"] = row.get(f"score_{criterion_name}", 0)
        if include_records:
            out_row["record_json"] = json.dumps(row.get("record"), separators=(",", ":"), ensure_ascii=True)
        writer.writerow(out_row)
    return buf.getvalue()


def _render_batch_artifact_export_jsonl(run: Dict[str, Any], include_records: bool) -> str:
    rows = _build_batch_export_rows(run, include_records=include_records)
    lines: List[str] = []
    for row in rows:
        payload = {
            "run_id": run["run_id"],
            "dataset_id": run.get("dataset_id"),
        }
        payload.update(row)
        lines.append(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
    return "\n".join(lines)


def _mark_batch_eval_failed(run_id: str, message: str) -> None:
    run_snapshot: Optional[Dict[str, Any]] = None
    with BATCH_EVAL_LOCK:
        run = BATCH_EVAL_RUNS.get(run_id)
        if run is None:
            return
        _finalize_batch_run_as_failed(run, message=message, details={"status": "failed"})
        run_snapshot = dict(run)
    if run_snapshot is not None:
        _persist_batch_eval_run_record(run_snapshot)


def _run_batch_eval_worker(run_id: str) -> None:
    try:
        run_snapshot: Optional[Dict[str, Any]] = None
        with BATCH_EVAL_LOCK:
            run = BATCH_EVAL_RUNS.get(run_id)
            if run is None or run.get("status") != "queued":
                return
            run["status"] = "running"
            run["started_at"] = _now_ts()
            run["updated_at"] = _now_ts()
            _append_batch_event(
                run,
                event_type="running",
                details={"status": "running"},
            )
            dataset_id = run["dataset_id"]
            criteria = list(run.get("criteria", []))
            model_name = run.get("model", MODEL_NAME)
            run_snapshot = dict(run)
        if run_snapshot is not None:
            _persist_batch_eval_run_record(run_snapshot)

        dataset = DATASETS.get(dataset_id)
        if dataset is None:
            dataset = _sqlite_dataset_row(dataset_id)
        if dataset is None:
            _mark_batch_eval_failed(run_id, "dataset_id not found")
            return

        records = list(dataset.get("records", []))
        total = len(records)
        record_scores: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        failed_records = 0
        max_retries = max(0, int(BATCH_EVAL_MAX_RETRIES))
        backoff_s = max(0.0, min(float(BATCH_EVAL_RETRY_BACKOFF_MS) / 1000.0, 10.0))
        retry_stats = {"attempted_records": 0, "total_retries": 0, "exhausted_records": 0}
        retry_log: List[Dict[str, Any]] = []

        for idx, record in enumerate(records):
            with BATCH_EVAL_LOCK:
                run = BATCH_EVAL_RUNS.get(run_id)
                if run is None:
                    return
                if run.get("cancel_requested"):
                    _finalize_batch_run_as_failed(
                        run,
                        message="batch run cancelled by user",
                        details={"cancelled": True, "at_record_index": idx},
                    )
                    run_snapshot = dict(run)
                    _persist_batch_eval_run_record(run_snapshot)
                    return

            attempt = 0
            row_scores: Dict[str, int] = {}
            scoring_error: str | None = None
            while attempt <= max_retries:
                attempt += 1
                try:
                    row_scores = {
                        criterion: _score_record_with_retry(record=record, criterion=criterion, attempt=attempt, record_index=idx)
                        for criterion in criteria
                    }
                    break
                except Exception as exc:
                    scoring_error = str(exc)
                    if attempt <= max_retries:
                        retry_stats["attempted_records"] += 1
                        retry_stats["total_retries"] += 1
                        retry_item = {
                            "record_index": idx,
                            "attempt": attempt,
                            "error": scoring_error,
                            "ts": _now_ts(),
                        }
                        retry_log.append(retry_item)
                        _append_batch_event(
                            run,
                            event_type="retry",
                            details={"record_index": idx, "attempt": attempt, "error": scoring_error},
                        )
                        if backoff_s > 0:
                            time.sleep(backoff_s)
                        continue
                    retry_stats["exhausted_records"] += 1
                    row_scores = {criterion: 0 for criterion in criteria}
                    break

            record_scores.append({"record_index": idx, "record": record, "scores": row_scores, "attempts": attempt})

            row_failed = False
            if scoring_error and all(v == 0 for v in row_scores.values()):
                row_failed = True
                failures.append(
                    {
                        "record_index": idx,
                        "criterion": "runtime",
                        "score": 0,
                        "reason": f"scoring failed after {attempt} attempt(s): {scoring_error}",
                        "record": record,
                    }
                )
            for criterion_name, score in row_scores.items():
                if score <= 3:
                    row_failed = True
                    failures.append(
                        {
                            "record_index": idx,
                            "criterion": criterion_name,
                            "score": score,
                            "reason": f"low score for {criterion_name}",
                            "record": record,
                        }
                    )
            if row_failed:
                failed_records += 1

            with BATCH_EVAL_LOCK:
                run = BATCH_EVAL_RUNS.get(run_id)
                if run is None:
                    return
                run["record_scores"] = list(record_scores)
                run["failures"] = list(failures)
                run["retries"] = list(retry_log[-1000:])
                run["retry_stats"] = dict(retry_stats)
                run["progress"] = _build_batch_progress(total=total, completed=idx + 1, failed=failed_records)
                run["updated_at"] = _now_ts()
                run_snapshot = dict(run)
            if idx == total - 1 or ((idx + 1) % 10 == 0):
                _append_batch_event(
                    run_snapshot,
                    event_type="progress",
                    details={"completed": idx + 1, "failed": failed_records, "total": total},
                )
                _persist_batch_eval_run_record(run_snapshot)

        summary = _compute_batch_summary(records, criteria)
        summary["retry_stats"] = dict(retry_stats)
        now = _now_ts()
        with BATCH_EVAL_LOCK:
            run = BATCH_EVAL_RUNS.get(run_id)
            if run is None:
                return
            run["status"] = "completed"
            run["summary"] = summary
            run["retries"] = list(retry_log[-1000:])
            run["retry_stats"] = dict(retry_stats)
            run["model"] = model_name
            run["progress"] = _build_batch_progress(total=total, completed=total, failed=summary.get("failed_items", 0))
            run["updated_at"] = now
            run["completed_at"] = now
            _append_batch_event(
                run,
                event_type="completed",
                details={
                    "status": "completed",
                    "total": total,
                    "failed_items": summary.get("failed_items", 0),
                    "total_retries": retry_stats.get("total_retries", 0),
                },
            )
            run_snapshot = dict(run)
        if run_snapshot is not None:
            _persist_batch_eval_run_record(run_snapshot)
    finally:
        _dispatch_batch_workers()


def _cancel_batch_eval_run_record(run_id: str) -> Optional[Dict[str, Any]]:
    with BATCH_EVAL_LOCK:
        run = BATCH_EVAL_RUNS.get(run_id)
        if run is None:
            return None
        status = str(run.get("status", ""))
        if status in {"completed", "failed"}:
            return dict(run)
        if status == "queued":
            run["cancel_requested"] = True
            _append_batch_event(run, event_type="cancel_requested", details={"status": status})
            _finalize_batch_run_as_failed(run, message="batch run cancelled before execution", details={"cancelled": True})
            return dict(run)
        run["cancel_requested"] = True
        run["updated_at"] = _now_ts()
        _append_batch_event(run, event_type="cancel_requested", details={"status": status})
        return dict(run)

def _is_protected_path(path: str) -> bool:
    if path in {"/", "/health", "/healthz", "/readyz", "/model"}:
        return False
    if path.startswith("/static"):
        return False
    return True


def _apply_rate_limit(bucket_key: str) -> Optional[Dict[str, Any]]:
    now = _now_ts()
    window_start = now - (now % 60)

    entry = RATE_LIMIT_BUCKETS.get(bucket_key)
    if entry is None or entry["window_start"] != window_start:
        entry = {"window_start": window_start, "count": 0}

    entry["count"] += 1
    RATE_LIMIT_BUCKETS[bucket_key] = entry

    remaining = max(RATE_LIMIT_PER_MINUTE - entry["count"], 0)
    reset_at = window_start + 60

    return {
        "limited": entry["count"] > RATE_LIMIT_PER_MINUTE,
        "limit": RATE_LIMIT_PER_MINUTE,
        "remaining": remaining,
        "reset_at": reset_at,
    }


def _rate_limit_headers(meta: Dict[str, Any]) -> Dict[str, str]:
    return {
        "X-RateLimit-Limit": str(meta["limit"]),
        "X-RateLimit-Remaining": str(meta["remaining"]),
        "X-RateLimit-Reset": str(meta["reset_at"]),
    }


def _chat_concurrency_snapshot() -> Dict[str, int]:
    with CHAT_CONCURRENCY_LOCK:
        active = int(CHAT_ACTIVE_REQUESTS)
    return {
        "active_requests": active,
        "max_concurrent_requests": max(1, int(CHAT_MAX_CONCURRENT_REQUESTS)),
    }


def _chat_try_acquire_slot() -> Tuple[bool, Dict[str, int]]:
    global CHAT_ACTIVE_REQUESTS
    with CHAT_CONCURRENCY_LOCK:
        active = int(CHAT_ACTIVE_REQUESTS)
        max_concurrent = max(1, int(CHAT_MAX_CONCURRENT_REQUESTS))
        if active >= max_concurrent:
            METRICS["chat_backpressure_rejections"] = int(METRICS.get("chat_backpressure_rejections", 0)) + 1
            return False, {"active_requests": active, "max_concurrent_requests": max_concurrent}
        CHAT_ACTIVE_REQUESTS = active + 1
        return True, {"active_requests": CHAT_ACTIVE_REQUESTS, "max_concurrent_requests": max_concurrent}


def _chat_release_slot() -> None:
    global CHAT_ACTIVE_REQUESTS
    with CHAT_CONCURRENCY_LOCK:
        CHAT_ACTIVE_REQUESTS = max(0, int(CHAT_ACTIVE_REQUESTS) - 1)


def _chat_timeout_deadline() -> float:
    timeout_ms = max(50, int(CHAT_REQUEST_TIMEOUT_MS))
    return time.monotonic() + (float(timeout_ms) / 1000.0)


def _enforce_chat_timeout(deadline: float, stage: str) -> None:
    if time.monotonic() > deadline:
        METRICS["chat_timeouts"] = int(METRICS.get("chat_timeouts", 0)) + 1
        raise HTTPException(
            status_code=503,
            detail={
                "code": "service_unavailable",
                "message": "chat request exceeded timeout budget",
                "details": {"reason": "timeout", "stage": stage, "timeout_ms": max(50, int(CHAT_REQUEST_TIMEOUT_MS))},
            },
        )


def _circuit_breaker_reset() -> None:
    with CIRCUIT_BREAKER_LOCK:
        CIRCUIT_BREAKER["state"] = "closed"
        CIRCUIT_BREAKER["consecutive_failures"] = 0
        CIRCUIT_BREAKER["opened_at"] = None
        CIRCUIT_BREAKER["last_failure_at"] = None
        CIRCUIT_BREAKER["last_failure_reason"] = None
        CIRCUIT_BREAKER["half_open_trial_inflight"] = False
        CIRCUIT_BREAKER["manual_forced_open"] = False
        CIRCUIT_BREAKER["manual_reason"] = None
        CIRCUIT_BREAKER["manual_expires_at"] = None


def _circuit_breaker_snapshot_unlocked() -> Dict[str, Any]:
    state = str(CIRCUIT_BREAKER.get("state", "closed"))
    opened_at = CIRCUIT_BREAKER.get("opened_at")
    cooldown = max(1, int(CIRCUIT_BREAKER_COOLDOWN_SECONDS))
    retry_after = 0
    if state == "open" and isinstance(opened_at, int):
        retry_after = max(0, int(opened_at) + cooldown - _now_ts())
    manual_expires_at = CIRCUIT_BREAKER.get("manual_expires_at")
    manual_remaining = None
    if isinstance(manual_expires_at, int):
        manual_remaining = max(0, manual_expires_at - _now_ts())
    return {
        "enabled": bool(CIRCUIT_BREAKER_ENABLED),
        "state": state,
        "consecutive_failures": int(CIRCUIT_BREAKER.get("consecutive_failures", 0) or 0),
        "failure_threshold": max(1, int(CIRCUIT_BREAKER_FAILURE_THRESHOLD)),
        "cooldown_seconds": cooldown,
        "retry_after_seconds": retry_after,
        "opened_at": opened_at,
        "last_failure_at": CIRCUIT_BREAKER.get("last_failure_at"),
        "last_failure_reason": CIRCUIT_BREAKER.get("last_failure_reason"),
        "half_open_trial_inflight": bool(CIRCUIT_BREAKER.get("half_open_trial_inflight", False)),
        "manual_forced_open": bool(CIRCUIT_BREAKER.get("manual_forced_open", False)),
        "manual_reason": CIRCUIT_BREAKER.get("manual_reason"),
        "manual_expires_at": manual_expires_at,
        "manual_remaining_seconds": manual_remaining,
    }


def _circuit_breaker_clear_manual_override_unlocked() -> None:
    CIRCUIT_BREAKER["manual_forced_open"] = False
    CIRCUIT_BREAKER["manual_reason"] = None
    CIRCUIT_BREAKER["manual_expires_at"] = None


def _circuit_breaker_snapshot() -> Dict[str, Any]:
    with CIRCUIT_BREAKER_LOCK:
        return _circuit_breaker_snapshot_unlocked()


def _circuit_breaker_force_open(reason: str | None = None, duration_seconds: int | None = None) -> Dict[str, Any]:
    now = _now_ts()
    expires_at = None
    if duration_seconds is not None:
        expires_at = now + max(1, min(int(duration_seconds), 86_400))
    with CIRCUIT_BREAKER_LOCK:
        CIRCUIT_BREAKER["manual_forced_open"] = True
        CIRCUIT_BREAKER["manual_reason"] = (reason or "manual_override").strip()[:500] or "manual_override"
        CIRCUIT_BREAKER["manual_expires_at"] = expires_at
        CIRCUIT_BREAKER["state"] = "open"
        CIRCUIT_BREAKER["opened_at"] = now
        CIRCUIT_BREAKER["half_open_trial_inflight"] = False
        return _circuit_breaker_snapshot_unlocked()


def _circuit_breaker_manual_reset() -> Dict[str, Any]:
    _circuit_breaker_reset()
    return _circuit_breaker_snapshot()


def _circuit_breaker_try_acquire() -> Tuple[bool, Dict[str, Any]]:
    if not CIRCUIT_BREAKER_ENABLED:
        return True, _circuit_breaker_snapshot()

    now = _now_ts()
    threshold = max(1, int(CIRCUIT_BREAKER_FAILURE_THRESHOLD))
    cooldown = max(1, int(CIRCUIT_BREAKER_COOLDOWN_SECONDS))

    with CIRCUIT_BREAKER_LOCK:
        if bool(CIRCUIT_BREAKER.get("manual_forced_open", False)):
            manual_expires_at = CIRCUIT_BREAKER.get("manual_expires_at")
            if isinstance(manual_expires_at, int) and now >= manual_expires_at:
                _circuit_breaker_clear_manual_override_unlocked()
                CIRCUIT_BREAKER["state"] = "closed"
                CIRCUIT_BREAKER["opened_at"] = None
                CIRCUIT_BREAKER["half_open_trial_inflight"] = False
                CIRCUIT_BREAKER["consecutive_failures"] = 0
            else:
                snap = _circuit_breaker_snapshot_unlocked()
                manual_remaining = snap.get("manual_remaining_seconds")
                if isinstance(manual_remaining, int):
                    snap["retry_after_seconds"] = max(1, manual_remaining)
                else:
                    snap["retry_after_seconds"] = max(1, int(snap.get("retry_after_seconds", 1)))
                return False, snap

        state = str(CIRCUIT_BREAKER.get("state", "closed"))
        if state == "open":
            opened_at = CIRCUIT_BREAKER.get("opened_at")
            if isinstance(opened_at, int) and (now - opened_at) >= cooldown:
                CIRCUIT_BREAKER["state"] = "half_open"
                CIRCUIT_BREAKER["half_open_trial_inflight"] = False
                state = "half_open"
            else:
                retry_after = max(1, (int(opened_at or now) + cooldown) - now)
                snap = _circuit_breaker_snapshot_unlocked()
                snap["retry_after_seconds"] = retry_after
                return False, snap

        if state == "half_open":
            if bool(CIRCUIT_BREAKER.get("half_open_trial_inflight", False)):
                snap = _circuit_breaker_snapshot_unlocked()
                snap["retry_after_seconds"] = 1
                return False, snap
            CIRCUIT_BREAKER["half_open_trial_inflight"] = True
            return True, _circuit_breaker_snapshot_unlocked()

        # closed
        CIRCUIT_BREAKER["state"] = "closed"
        if int(CIRCUIT_BREAKER.get("consecutive_failures", 0) or 0) >= threshold:
            CIRCUIT_BREAKER["consecutive_failures"] = 0
        return True, _circuit_breaker_snapshot_unlocked()


def _circuit_breaker_record_success() -> None:
    if not CIRCUIT_BREAKER_ENABLED:
        return
    with CIRCUIT_BREAKER_LOCK:
        CIRCUIT_BREAKER["consecutive_failures"] = 0
        CIRCUIT_BREAKER["half_open_trial_inflight"] = False
        CIRCUIT_BREAKER["state"] = "closed"
        CIRCUIT_BREAKER["opened_at"] = None


def _circuit_breaker_release_trial() -> None:
    if not CIRCUIT_BREAKER_ENABLED:
        return
    with CIRCUIT_BREAKER_LOCK:
        if str(CIRCUIT_BREAKER.get("state", "closed")) == "half_open":
            CIRCUIT_BREAKER["half_open_trial_inflight"] = False


def _circuit_breaker_record_failure(reason: str) -> None:
    if not CIRCUIT_BREAKER_ENABLED:
        return
    now = _now_ts()
    threshold = max(1, int(CIRCUIT_BREAKER_FAILURE_THRESHOLD))
    with CIRCUIT_BREAKER_LOCK:
        state = str(CIRCUIT_BREAKER.get("state", "closed"))
        CIRCUIT_BREAKER["last_failure_at"] = now
        CIRCUIT_BREAKER["last_failure_reason"] = reason[:500]
        CIRCUIT_BREAKER["half_open_trial_inflight"] = False
        if state == "half_open":
            CIRCUIT_BREAKER["state"] = "open"
            CIRCUIT_BREAKER["opened_at"] = now
            CIRCUIT_BREAKER["consecutive_failures"] = threshold
            return

        failures = int(CIRCUIT_BREAKER.get("consecutive_failures", 0) or 0) + 1
        CIRCUIT_BREAKER["consecutive_failures"] = failures
        if failures >= threshold:
            CIRCUIT_BREAKER["state"] = "open"
            CIRCUIT_BREAKER["opened_at"] = now


def _is_circuit_breaker_failure(exc: Exception) -> Tuple[bool, str]:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if exc.status_code >= 500:
            return True, f"http_{exc.status_code}"
        if isinstance(detail, dict):
            code = str(detail.get("code", "") or detail.get("error", ""))
            if code in {"service_unavailable", "generation_failed"}:
                return True, code
        return False, f"http_{exc.status_code}"
    return True, exc.__class__.__name__


def _runtime_config_snapshot() -> Dict[str, Any]:
    return {
        "rate_limit_per_minute": int(RATE_LIMIT_PER_MINUTE),
        "chat_max_concurrent_requests": max(1, int(CHAT_MAX_CONCURRENT_REQUESTS)),
        "chat_request_timeout_ms": max(50, int(CHAT_REQUEST_TIMEOUT_MS)),
        "circuit_breaker_enabled": bool(CIRCUIT_BREAKER_ENABLED),
        "circuit_breaker_failure_threshold": max(1, int(CIRCUIT_BREAKER_FAILURE_THRESHOLD)),
        "circuit_breaker_cooldown_seconds": max(1, int(CIRCUIT_BREAKER_COOLDOWN_SECONDS)),
        "batch_eval_max_concurrent_runs": max(1, int(BATCH_EVAL_MAX_CONCURRENT_RUNS)),
        "batch_eval_max_retries": max(0, int(BATCH_EVAL_MAX_RETRIES)),
        "batch_eval_retry_backoff_ms": max(0, int(BATCH_EVAL_RETRY_BACKOFF_MS)),
        "slo_window_seconds": max(60, int(SLO_WINDOW_SECONDS)),
        "slo_error_budget_pct": max(0.01, float(SLO_ERROR_BUDGET_PCT)),
    }


def _validate_runtime_config_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "payload must be an object"})

    allowed_keys = {
        "rate_limit_per_minute",
        "chat_max_concurrent_requests",
        "chat_request_timeout_ms",
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_cooldown_seconds",
        "batch_eval_max_concurrent_runs",
        "batch_eval_max_retries",
        "batch_eval_retry_backoff_ms",
        "slo_window_seconds",
        "slo_error_budget_pct",
    }
    unknown = [k for k in patch.keys() if k not in allowed_keys]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": "unknown runtime config key(s)", "details": {"unknown_keys": unknown}},
        )

    validated: Dict[str, Any] = {}
    for key, value in patch.items():
        if key == "slo_error_budget_pct":
            if not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_request", "message": f"{key} must be number"},
                )
            fv = float(value)
            if fv < 0.01 or fv > 100.0:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (0.01..100.0)"})
            validated[key] = fv
            continue

        if key == "circuit_breaker_enabled":
            if not isinstance(value, bool):
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_request", "message": f"{key} must be boolean"},
                )
            validated[key] = value
            continue

        if not isinstance(value, int):
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_request", "message": f"{key} must be integer"},
            )

        if key == "rate_limit_per_minute":
            if value < 1 or value > 100_000:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (1..100000)"})
        elif key == "chat_max_concurrent_requests":
            if value < 1 or value > 10_000:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (1..10000)"})
        elif key == "chat_request_timeout_ms":
            if value < 50 or value > 600_000:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (50..600000)"})
        elif key == "circuit_breaker_failure_threshold":
            if value < 1 or value > 10_000:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (1..10000)"})
        elif key == "circuit_breaker_cooldown_seconds":
            if value < 1 or value > 86_400:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (1..86400)"})
        elif key == "batch_eval_max_concurrent_runs":
            if value < 1 or value > 1_000:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (1..1000)"})
        elif key == "batch_eval_max_retries":
            if value < 0 or value > 100:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (0..100)"})
        elif key == "batch_eval_retry_backoff_ms":
            if value < 0 or value > 60_000:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (0..60000)"})
        elif key == "slo_window_seconds":
            if value < 60 or value > 86_400:
                raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"{key} out of range (60..86400)"})

        validated[key] = value

    return validated


def _apply_runtime_config_patch(validated_patch: Dict[str, Any]) -> Dict[str, Any]:
    global RATE_LIMIT_PER_MINUTE
    global CHAT_MAX_CONCURRENT_REQUESTS
    global CHAT_REQUEST_TIMEOUT_MS
    global CIRCUIT_BREAKER_ENABLED
    global CIRCUIT_BREAKER_FAILURE_THRESHOLD
    global CIRCUIT_BREAKER_COOLDOWN_SECONDS
    global BATCH_EVAL_MAX_CONCURRENT_RUNS
    global BATCH_EVAL_MAX_RETRIES
    global BATCH_EVAL_RETRY_BACKOFF_MS
    global SLO_WINDOW_SECONDS
    global SLO_ERROR_BUDGET_PCT

    for key, value in validated_patch.items():
        if key == "rate_limit_per_minute":
            RATE_LIMIT_PER_MINUTE = int(value)
        elif key == "chat_max_concurrent_requests":
            CHAT_MAX_CONCURRENT_REQUESTS = int(value)
        elif key == "chat_request_timeout_ms":
            CHAT_REQUEST_TIMEOUT_MS = int(value)
        elif key == "circuit_breaker_enabled":
            CIRCUIT_BREAKER_ENABLED = bool(value)
            if not CIRCUIT_BREAKER_ENABLED:
                _circuit_breaker_reset()
        elif key == "circuit_breaker_failure_threshold":
            CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(value)
        elif key == "circuit_breaker_cooldown_seconds":
            CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(value)
        elif key == "batch_eval_max_concurrent_runs":
            BATCH_EVAL_MAX_CONCURRENT_RUNS = int(value)
            _dispatch_batch_workers()
        elif key == "batch_eval_max_retries":
            BATCH_EVAL_MAX_RETRIES = int(value)
        elif key == "batch_eval_retry_backoff_ms":
            BATCH_EVAL_RETRY_BACKOFF_MS = int(value)
        elif key == "slo_window_seconds":
            SLO_WINDOW_SECONDS = int(value)
        elif key == "slo_error_budget_pct":
            SLO_ERROR_BUDGET_PCT = float(value)

    return _runtime_config_snapshot()


def _validate_runtime_profile_name(profile_name: str) -> str:
    name = str(profile_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "profile_name must be non-empty"})
    if len(name) > 120:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "profile_name too long (max 120)"})
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(ch not in allowed for ch in name):
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": "profile_name contains invalid characters"},
        )
    return name


def _persist_runtime_profiles_if_needed() -> None:
    if STATE_PERSISTENCE_ENABLED:
        _save_state_to_disk()


def _validate_runbook_steps(raw_steps: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_steps, list) or not raw_steps:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "steps must be a non-empty array"})
    if len(raw_steps) > 200:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "steps length exceeds 200"})
    steps: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_steps, start=1):
        title = str(item or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": f"step {idx} title must be non-empty"})
        steps.append(
            {
                "step_id": f"step_{idx}",
                "title": title[:300],
                "status": "pending",
                "note": None,
                "updated_at": _now_ts(),
                "completed_at": None,
            }
        )
    return steps


def _get_runbook_or_404(runbook_id: str) -> Dict[str, Any]:
    runbook = RUNBOOK_RUNS.get(str(runbook_id))
    if not isinstance(runbook, dict):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runbook not found"})
    return runbook


def _validate_runbook_template_id(template_id: str) -> str:
    tid = str(template_id or "").strip()
    if not tid:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "template_id must be non-empty"})
    if len(tid) > 120:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "template_id too long (max 120)"})
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(ch not in allowed for ch in tid):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "template_id contains invalid characters"})
    return tid


def _maintenance_snapshot() -> Dict[str, Any]:
    with MAINTENANCE_LOCK:
        return dict(MAINTENANCE_STATE)


def _maintenance_disable() -> Dict[str, Any]:
    with MAINTENANCE_LOCK:
        MAINTENANCE_STATE["active"] = False
        MAINTENANCE_STATE["reason"] = None
        MAINTENANCE_STATE["enabled_at"] = None
        MAINTENANCE_STATE["expires_at"] = None
        MAINTENANCE_STATE["read_only"] = False
        return dict(MAINTENANCE_STATE)


def _maintenance_enable(reason: str | None = None, duration_seconds: int | None = None, read_only: bool = False) -> Dict[str, Any]:
    now = _now_ts()
    expires_at = None
    if duration_seconds is not None:
        expires_at = now + max(1, min(int(duration_seconds), 86_400))
    with MAINTENANCE_LOCK:
        MAINTENANCE_STATE["active"] = True
        MAINTENANCE_STATE["reason"] = (reason or "maintenance").strip()[:500] or "maintenance"
        MAINTENANCE_STATE["enabled_at"] = now
        MAINTENANCE_STATE["expires_at"] = expires_at
        MAINTENANCE_STATE["read_only"] = bool(read_only)
        return dict(MAINTENANCE_STATE)


def _maintenance_is_active() -> bool:
    now = _now_ts()
    with MAINTENANCE_LOCK:
        active = bool(MAINTENANCE_STATE.get("active", False))
        if not active:
            return False
        expires_at = MAINTENANCE_STATE.get("expires_at")
        if isinstance(expires_at, int) and now >= expires_at:
            MAINTENANCE_STATE["active"] = False
            MAINTENANCE_STATE["reason"] = None
            MAINTENANCE_STATE["enabled_at"] = None
            MAINTENANCE_STATE["expires_at"] = None
            MAINTENANCE_STATE["read_only"] = False
            return False
        return True


def _maintenance_is_exempt_path(path: str) -> bool:
    if path in {"/", "/health", "/healthz", "/readyz", "/model"}:
        return True
    if path.startswith("/static"):
        return True
    if path.startswith("/v1/admin/maintenance"):
        return True
    if path.startswith("/v1/admin"):
        return True
    return False


def _maintenance_should_block_request(path: str, method: str, is_admin: bool) -> Tuple[bool, Dict[str, Any]]:
    if not _maintenance_is_active():
        return False, {}
    if is_admin:
        return False, {}
    if _maintenance_is_exempt_path(path):
        return False, {}

    snap = _maintenance_snapshot()
    read_only = bool(snap.get("read_only", False))
    if read_only and method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return False, {}

    retry_after = 0
    expires_at = snap.get("expires_at")
    if isinstance(expires_at, int):
        retry_after = max(1, expires_at - _now_ts())
    details = {
        "reason": snap.get("reason") or "maintenance",
        "read_only": read_only,
        "retry_after_seconds": retry_after if retry_after > 0 else None,
    }
    return True, details


def _record_slo_event(status_code: int, path: str) -> None:
    now = _now_ts()
    with SLO_LOCK:
        SLO_EVENTS.append({"ts": now, "status_code": int(status_code), "path": path})
        max_window = max(60, int(SLO_WINDOW_SECONDS))
        cutoff = now - max_window
        if len(SLO_EVENTS) > 10_000:
            del SLO_EVENTS[:-10_000]
        while SLO_EVENTS and int(SLO_EVENTS[0].get("ts", 0) or 0) < cutoff:
            del SLO_EVENTS[0]
    _update_slo_incident_state()


def _trim_slo_incidents() -> None:
    max_len = max(10, int(SLO_INCIDENT_HISTORY_LIMIT))
    if len(SLO_INCIDENTS) > max_len:
        del SLO_INCIDENTS[:-max_len]


def _slo_snapshot(window_seconds: int | None = None) -> Dict[str, Any]:
    now = _now_ts()
    configured_window = max(60, int(SLO_WINDOW_SECONDS))
    window = configured_window if window_seconds is None else max(60, min(int(window_seconds), 86_400))
    budget_pct = max(0.01, float(SLO_ERROR_BUDGET_PCT))
    cutoff = now - window
    with SLO_LOCK:
        rows = [e for e in SLO_EVENTS if int(e.get("ts", 0) or 0) >= cutoff]
    total = len(rows)
    failed = sum(1 for e in rows if int(e.get("status_code", 0) or 0) >= 500)
    error_rate_pct = round((failed / total) * 100.0, 4) if total else 0.0
    remaining_pct = round(max(0.0, budget_pct - error_rate_pct), 4)
    burn_rate = round((error_rate_pct / budget_pct), 4) if budget_pct > 0 else 0.0
    return {
        "window_seconds": window,
        "configured_window_seconds": configured_window,
        "error_budget_pct": budget_pct,
        "requests_total": total,
        "failed_requests": failed,
        "error_rate_pct": error_rate_pct,
        "error_budget_remaining_pct": remaining_pct,
        "burn_rate": burn_rate,
        "breached": error_rate_pct > budget_pct if total > 0 else False,
    }


def _update_slo_incident_state() -> None:
    now = _now_ts()
    snap = _slo_snapshot()
    breached_now = bool(snap.get("breached", False))
    with SLO_LOCK:
        was_breached = bool(SLO_STATE.get("breached", False))
        current_incident_id = SLO_STATE.get("current_incident_id")

        if breached_now and not was_breached:
            incident_id = f"sloi_{uuid.uuid4().hex[:10]}"
            SLO_INCIDENTS.append(
                {
                    "incident_id": incident_id,
                    "status": "open",
                    "opened_at": now,
                    "updated_at": now,
                    "resolved_at": None,
                    "acknowledged_at": None,
                    "acknowledged_by": None,
                    "peak_error_rate_pct": float(snap.get("error_rate_pct", 0.0)),
                    "latest_error_rate_pct": float(snap.get("error_rate_pct", 0.0)),
                    "error_budget_pct": float(snap.get("error_budget_pct", 0.0)),
                    "window_seconds": int(snap.get("window_seconds", 0) or 0),
                    "request_count_at_open": int(snap.get("requests_total", 0) or 0),
                    "failed_count_at_open": int(snap.get("failed_requests", 0) or 0),
                    "notes": [],
                }
            )
            _trim_slo_incidents()
            SLO_STATE["breached"] = True
            SLO_STATE["current_incident_id"] = incident_id
            return

        if breached_now and was_breached:
            if current_incident_id:
                for item in reversed(SLO_INCIDENTS):
                    if str(item.get("incident_id")) != str(current_incident_id):
                        continue
                    item["updated_at"] = now
                    item["latest_error_rate_pct"] = float(snap.get("error_rate_pct", 0.0))
                    item["peak_error_rate_pct"] = max(
                        float(item.get("peak_error_rate_pct", 0.0)),
                        float(snap.get("error_rate_pct", 0.0)),
                    )
                    break
            return

        if (not breached_now) and was_breached:
            if current_incident_id:
                for item in reversed(SLO_INCIDENTS):
                    if str(item.get("incident_id")) != str(current_incident_id):
                        continue
                    item["status"] = "resolved"
                    item["updated_at"] = now
                    item["resolved_at"] = now
                    item["latest_error_rate_pct"] = float(snap.get("error_rate_pct", 0.0))
                    break
            SLO_STATE["breached"] = False
            SLO_STATE["current_incident_id"] = None
            _trim_slo_incidents()
            return

        # steady not-breached
        SLO_STATE["breached"] = False
        SLO_STATE["current_incident_id"] = None


def _find_slo_incident(incident_id: str) -> Optional[Dict[str, Any]]:
    target = str(incident_id or "").strip()
    if not target:
        return None
    with SLO_LOCK:
        for item in reversed(SLO_INCIDENTS):
            if str(item.get("incident_id", "")) == target:
                return item
    return None


def _render_metrics_text() -> str:
    cache = _cache_stats()
    counts = _state_counts()
    cb = _circuit_breaker_snapshot()
    cb_state = 1 if cb["state"] == "open" else (2 if cb["state"] == "half_open" else 0)
    chat_runtime = _chat_concurrency_snapshot()
    maintenance = _maintenance_snapshot()
    maintenance_active = 1 if _maintenance_is_active() else 0
    slo = _slo_snapshot()
    lines = [
        "# HELP http_requests_total Total chat requests observed by this process",
        "# TYPE http_requests_total counter",
        f"http_requests_total {METRICS['chat_requests']}",
        "# HELP cache_hits_total Total chat cache hits",
        "# TYPE cache_hits_total counter",
        f"cache_hits_total {cache['hits']}",
        "# HELP cache_misses_total Total chat cache misses",
        "# TYPE cache_misses_total counter",
        f"cache_misses_total {cache['misses']}",
        "# HELP cache_hit_ratio Chat cache hit ratio",
        "# TYPE cache_hit_ratio gauge",
        f"cache_hit_ratio {cache['hit_rate']}",
        "# HELP datasets_total Number of datasets stored",
        "# TYPE datasets_total gauge",
        f"datasets_total {counts['datasets']}",
        "# HELP eval_runs_total Number of eval runs stored",
        "# TYPE eval_runs_total gauge",
        f"eval_runs_total {counts['eval_runs']}",
        "# HELP batch_eval_runs_total Number of batch eval runs stored",
        "# TYPE batch_eval_runs_total gauge",
        f"batch_eval_runs_total {counts['batch_eval_runs']}",
        "# HELP runbook_runs_total Number of runbook runs stored",
        "# TYPE runbook_runs_total gauge",
        f"runbook_runs_total {counts['runbook_runs']}",
        "# HELP runbook_templates_total Number of runbook templates stored",
        "# TYPE runbook_templates_total gauge",
        f"runbook_templates_total {counts['runbook_templates']}",
        "# HELP rate_limit_rejections_total Number of rate-limited requests",
        "# TYPE rate_limit_rejections_total counter",
        f"rate_limit_rejections_total {sum(max(v['count'] - RATE_LIMIT_PER_MINUTE, 0) for v in RATE_LIMIT_BUCKETS.values())}",
        "# HELP audit_logs_total Number of audit log entries stored",
        "# TYPE audit_logs_total gauge",
        f"audit_logs_total {counts['audit_logs']}",
        "# HELP circuit_breaker_state Inference circuit breaker state (0=closed,1=open,2=half_open)",
        "# TYPE circuit_breaker_state gauge",
        f"circuit_breaker_state {cb_state}",
        "# HELP circuit_breaker_consecutive_failures Consecutive inference failures tracked by circuit breaker",
        "# TYPE circuit_breaker_consecutive_failures gauge",
        f"circuit_breaker_consecutive_failures {cb['consecutive_failures']}",
        "# HELP chat_active_requests In-flight chat requests currently being processed",
        "# TYPE chat_active_requests gauge",
        f"chat_active_requests {chat_runtime['active_requests']}",
        "# HELP chat_backpressure_rejections_total Chat requests rejected due to concurrency backpressure",
        "# TYPE chat_backpressure_rejections_total counter",
        f"chat_backpressure_rejections_total {int(METRICS.get('chat_backpressure_rejections', 0))}",
        "# HELP chat_timeouts_total Chat requests that exceeded timeout budget",
        "# TYPE chat_timeouts_total counter",
        f"chat_timeouts_total {int(METRICS.get('chat_timeouts', 0))}",
        "# HELP maintenance_mode_active Whether maintenance mode is active (1=true,0=false)",
        "# TYPE maintenance_mode_active gauge",
        f"maintenance_mode_active {maintenance_active}",
        "# HELP maintenance_rejections_total Requests rejected due to maintenance mode",
        "# TYPE maintenance_rejections_total counter",
        f"maintenance_rejections_total {int(METRICS.get('maintenance_rejections', 0))}",
        "# HELP slo_error_rate_pct Rolling SLO error rate percentage",
        "# TYPE slo_error_rate_pct gauge",
        f"slo_error_rate_pct {slo['error_rate_pct']}",
        "# HELP slo_error_budget_breached Whether SLO error budget is breached (1=true,0=false)",
        "# TYPE slo_error_budget_breached gauge",
        f"slo_error_budget_breached {1 if slo['breached'] else 0}",
    ]
    return "\n".join(lines) + "\n"


def _startup_impl() -> None:
    global tokenizer, model, MODEL_META, RETENTION_THREAD

    _init_cache_backend()
    _init_auth_registry()
    _init_rag_vector_backend()
    _init_tracing_backend()
    RETENTION_STOP_EVENT.clear()

    log_json(LOGGER, {"event": "startup", "model_name": MODEL_NAME, "device": DEVICE})

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        missing = []
        if not TORCH_AVAILABLE:
            missing.append("torch")
        if not TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        raise RuntimeError(f"Missing required runtime dependencies: {', '.join(missing)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()

        context_window = _get_context_window()
        heads, hidden = _get_heads_and_hidden()

        MODEL_META = {
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "model_type": _get_model_type(),
            "attention_masking": _get_attention_masking(),
            "context_window": context_window,
            "attention_heads": heads,
            "hidden_size": hidden,
        }

        log_json(LOGGER, {"event": "model_loaded", **MODEL_META})

    except Exception as e:
        log_json(LOGGER, {"event": "startup_failed", "error": repr(e)})
        raise

    if RETENTION_SWEEP_ENABLED and SOFT_DELETE_RETENTION_SECONDS >= 0:
        RETENTION_THREAD = threading.Thread(target=_retention_sweep_loop, daemon=True, name="retention-sweep")
        RETENTION_THREAD.start()


def _shutdown_impl() -> None:
    global RETENTION_THREAD
    RETENTION_STOP_EVENT.set()
    if RETENTION_THREAD is not None and RETENTION_THREAD.is_alive():
        RETENTION_THREAD.join(timeout=2.0)
    RETENTION_THREAD = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    _startup_impl()
    try:
        yield
    finally:
        _shutdown_impl()


app = FastAPI(title="LLM Inference Server", version="0.3.0", lifespan=app_lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

def _error_response(status_code: int, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    rid = REQUEST_ID.get()
    content: Dict[str, Any] = {
        "error": {"code": code, "message": message},
        "request_id": rid,
    }
    if details:
        content["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=content)


def _require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes([], x_api_key=x_api_key, authorization=authorization)


def _require_chat_invoke(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["chat:invoke"], x_api_key=x_api_key, authorization=authorization)


def _require_agent_invoke(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["agent:invoke"], x_api_key=x_api_key, authorization=authorization)


def _require_evaluate_invoke(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["evaluate:invoke"], x_api_key=x_api_key, authorization=authorization)


def _require_embeddings_invoke(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["embeddings:invoke"], x_api_key=x_api_key, authorization=authorization)


def _require_rag_query(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["rag:query"], x_api_key=x_api_key, authorization=authorization)


def _require_rag_read(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["rag:read"], x_api_key=x_api_key, authorization=authorization)


def _require_rag_write(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["rag:write"], x_api_key=x_api_key, authorization=authorization)


def _require_datasets_read(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["datasets:read"], x_api_key=x_api_key, authorization=authorization)


def _require_datasets_write(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["datasets:write"], x_api_key=x_api_key, authorization=authorization)


def _require_evals_read(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["evals:read"], x_api_key=x_api_key, authorization=authorization)


def _require_evals_write(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["evals:write"], x_api_key=x_api_key, authorization=authorization)


def _require_batch_read(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["batch:read"], x_api_key=x_api_key, authorization=authorization)


def _require_batch_write(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["batch:write"], x_api_key=x_api_key, authorization=authorization)


def _require_metrics_read(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _require_auth_for_scopes(["metrics:read"], x_api_key=x_api_key, authorization=authorization)


def _require_admin(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    auth = _require_auth_for_scopes([], x_api_key=x_api_key, authorization=authorization)
    role = str(auth.get("role", "")).strip().lower()
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail={"code": "forbidden", "message": "Admin role required"},
        )


def _agent_tool_allowed(auth: Dict[str, Any], tool_name: str) -> bool:
    meta = AGENT_TOOL_CATALOG.get(tool_name, {})
    role = str(auth.get("role", "viewer")).strip().lower()
    scopes = [str(s).strip() for s in auth.get("scopes", []) if str(s).strip()]
    required_role = meta.get("required_role")
    if isinstance(required_role, str) and required_role.strip():
        if role != required_role.strip().lower():
            return False
    required_scope = meta.get("required_scope")
    if isinstance(required_scope, str) and required_scope.strip():
        if not _scope_allowed(scopes, required_scope.strip()):
            return False
    return True


def _agent_tool_rows(auth: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tool_name in sorted(AGENT_TOOL_CATALOG.keys()):
        meta = AGENT_TOOL_CATALOG.get(tool_name, {})
        rows.append(
            {
                "tool": tool_name,
                "allowed": _agent_tool_allowed(auth, tool_name),
                "description": str(meta.get("description", "")),
                "required_scope": meta.get("required_scope"),
                "required_role": meta.get("required_role"),
                "params": dict(meta.get("params", {})) if isinstance(meta.get("params"), dict) else {},
            }
        )
    return rows


def _execute_agent_tool(tool_name: str, args: Dict[str, Any], auth: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in AGENT_TOOL_CATALOG:
        return {"ok": False, "error": "unknown_tool"}
    if not _agent_tool_allowed(auth, tool_name):
        return {"ok": False, "error": "forbidden_tool"}

    if tool_name == "datasets.list":
        limit = max(1, min(int(args.get("limit", 10) or 10), 100))
        dataset_type = args.get("type")
        dataset_status = args.get("status")
        payload = v1_list_datasets(type=dataset_type, status=dataset_status, limit=limit, include_deleted=False)
        return {"ok": True, "data": payload.model_dump()}

    if tool_name == "metrics.dashboard":
        window = str(args.get("window", "15m") or "15m")
        return {"ok": True, "data": metrics_dashboard(window=window)}

    if tool_name == "rag.backend_status":
        return {"ok": True, "data": _rag_vector_backend_snapshot()}

    if tool_name == "tracing.status":
        return {"ok": True, "data": _tracing_backend_snapshot()}

    if tool_name == "runbooks.list":
        limit = max(1, min(int(args.get("limit", 10) or 10), 100))
        status = args.get("status")
        return {"ok": True, "data": v1_admin_runbooks(limit=limit, status=status)}

    return {"ok": False, "error": "not_implemented"}


def _plan_agent_tools(goal: str, available_tools: List[str]) -> List[Dict[str, Any]]:
    goal_lower = goal.lower()
    calls: List[Dict[str, Any]] = []

    def add(tool: str, args: Dict[str, Any] | None = None) -> None:
        if tool in available_tools and tool not in {c["tool"] for c in calls}:
            calls.append({"tool": tool, "args": args or {}})

    if "dataset" in goal_lower:
        add("datasets.list", {"limit": 10})
    if "metric" in goal_lower or "dashboard" in goal_lower or "kpi" in goal_lower:
        add("metrics.dashboard", {"window": "15m"})
    if "rag" in goal_lower or "vector" in goal_lower:
        add("rag.backend_status")
    if "tracing" in goal_lower or "trace" in goal_lower:
        add("tracing.status")
    if "runbook" in goal_lower:
        add("runbooks.list", {"limit": 10})
    if not calls:
        add("metrics.dashboard", {"window": "15m"})
    return calls


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/react")
def react_console():
    return FileResponse("app/static/react.html")


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    REQUEST_ID.set(rid)
    AUTH_CONTEXT.set(None)

    start = time.perf_counter()

    path = str(request.url.path)
    method = request.method.upper()
    is_protected = _is_protected_path(path)
    provided_key = _resolve_api_key(request.headers.get("x-api-key"), request.headers.get("authorization"))
    auth = _authenticate_api_key_value(provided_key)
    is_admin = bool(auth and str(auth.get("role", "")).strip().lower() == "admin")

    blocked, maintenance_details = _maintenance_should_block_request(path=path, method=method, is_admin=is_admin)
    if blocked:
        METRICS["maintenance_rejections"] = int(METRICS.get("maintenance_rejections", 0)) + 1
        response = _error_response(
            503,
            "service_unavailable",
            "service temporarily unavailable (maintenance mode)",
            maintenance_details,
        )
        retry_after = maintenance_details.get("retry_after_seconds")
        if isinstance(retry_after, int) and retry_after > 0:
            response.headers["Retry-After"] = str(retry_after)
        response.headers["x-request-id"] = rid
        response.headers["x-latency-ms"] = str(int((time.perf_counter() - start) * 1000))
        return response

    rate_meta = None
    if is_protected:
        bucket_key = provided_key or "anonymous"
        rate_meta = _apply_rate_limit(bucket_key)
        if rate_meta["limited"]:
            details = {
                "limit": rate_meta["limit"],
                "remaining": rate_meta["remaining"],
                "reset_at": _iso_utc_from_epoch(rate_meta["reset_at"]),
                "bucket": "api_key_per_minute",
            }
            response = _error_response(429, "rate_limited", "Rate limit exceeded", details)
            response.headers["Retry-After"] = str(max(rate_meta["reset_at"] - _now_ts(), 1))
            for h, v in _rate_limit_headers(rate_meta).items():
                response.headers[h] = v
            response.headers["x-request-id"] = rid
            response.headers["x-latency-ms"] = str(int((time.perf_counter() - start) * 1000))
            _record_slo_event(status_code=429, path=path)
            return response

    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_json(
            LOGGER,
            {
                "event": "unhandled_exception",
                "request_id": rid,
                "path": str(request.url.path),
                "method": request.method,
                "latency_ms": latency_ms,
                "error": repr(exc),
            },
        )
        raise

    latency_ms = int((time.perf_counter() - start) * 1000)
    response.headers["x-request-id"] = rid
    response.headers["x-latency-ms"] = str(latency_ms)
    if rate_meta is not None:
        for h, v in _rate_limit_headers(rate_meta).items():
            response.headers[h] = v
    if is_protected:
        _record_slo_event(status_code=int(getattr(response, "status_code", 200)), path=path)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = REQUEST_ID.get()
    log_json(
        LOGGER,
        {
            "event": "http_exception",
            "request_id": rid,
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        return _error_response(exc.status_code, exc.detail["code"], exc.detail["message"], exc.detail.get("details"))
    if isinstance(exc.detail, str):
        code = "invalid_request" if exc.status_code == 400 else "internal_error"
        return _error_response(exc.status_code, code, exc.detail)
    return _error_response(exc.status_code, "invalid_request", "Request failed", {"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error_response(
        400,
        "invalid_request",
        "Request validation failed",
        {"errors": exc.errors()},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "request_id": REQUEST_ID.get()}


@app.get("/v1/health")
def v1_health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "request_id": REQUEST_ID.get()}


@app.get("/v1/auth/context")
def v1_auth_context(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    rid = REQUEST_ID.get()
    auth = _authenticate_from_headers(x_api_key=x_api_key, authorization=authorization)
    role = str(auth.get("role", "viewer")).strip().lower() or "viewer"
    scopes = [str(s).strip() for s in auth.get("scopes", []) if str(s).strip()]
    capabilities = _build_auth_capabilities(auth)
    return {
        "data": {
            "role": role,
            "is_admin": role == "admin",
            "scopes": scopes,
            "scope_count": len(scopes),
            "capabilities": capabilities,
            "capability_count": sum(1 for allowed in capabilities.values() if allowed),
            "key_fingerprint": _audit_key_fingerprint(auth.get("api_key")),
        },
        "request_id": rid,
    }


@app.get("/v1/auth/capabilities")
def v1_auth_capabilities(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    rid = REQUEST_ID.get()
    auth = _authenticate_from_headers(x_api_key=x_api_key, authorization=authorization)
    capabilities = _build_auth_capabilities(auth)
    rows = []
    for key in sorted(AUTH_CAPABILITY_CATALOG.keys()):
        meta = AUTH_CAPABILITY_CATALOG.get(key, {})
        rows.append(
            {
                "capability": key,
                "allowed": bool(capabilities.get(key, False)),
                "description": str(meta.get("description", "")),
                "required_scope": meta.get("required_scope"),
                "required_role": meta.get("required_role"),
            }
        )
    return {"data": rows, "count": len(rows), "request_id": rid}


@app.get("/v1/agent/tools", dependencies=[Depends(_require_agent_invoke)])
def v1_agent_tools(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    rid = REQUEST_ID.get()
    auth = _authenticate_from_headers(x_api_key=x_api_key, authorization=authorization)
    rows = _agent_tool_rows(auth)
    return {"data": rows, "count": len(rows), "request_id": rid}


@app.post("/v1/agent/run", dependencies=[Depends(_require_agent_invoke)])
def v1_agent_run(
    payload: Dict[str, Any] = Body(...),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    rid = REQUEST_ID.get()
    auth = _authenticate_from_headers(x_api_key=x_api_key, authorization=authorization)
    goal = str(payload.get("goal", "") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "goal must be non-empty"})

    requested_tools = payload.get("requested_tools")
    tool_args_map = payload.get("tool_args")
    if not isinstance(tool_args_map, dict):
        tool_args_map = {}

    available_rows = _agent_tool_rows(auth)
    allowed_tools = [row["tool"] for row in available_rows if bool(row.get("allowed"))]

    calls: List[Dict[str, Any]] = []
    if isinstance(requested_tools, list) and requested_tools:
        seen = set()
        for item in requested_tools:
            tool = str(item or "").strip()
            if not tool or tool in seen:
                continue
            seen.add(tool)
            calls.append({"tool": tool, "args": dict(tool_args_map.get(tool, {})) if isinstance(tool_args_map.get(tool), dict) else {}})
    else:
        calls = _plan_agent_tools(goal=goal, available_tools=allowed_tools)
        for call in calls:
            tool = call["tool"]
            if isinstance(tool_args_map.get(tool), dict):
                merged = dict(call.get("args", {}))
                merged.update(tool_args_map.get(tool, {}))
                call["args"] = merged

    execution = []
    for call in calls:
        tool = str(call.get("tool", "")).strip()
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        result = _execute_agent_tool(tool, args, auth)
        execution.append({"tool": tool, "args": args, "result": result})

    succeeded = sum(1 for item in execution if bool(item.get("result", {}).get("ok")))
    failed = len(execution) - succeeded
    return {
        "goal": goal,
        "planned_calls": calls,
        "execution": execution,
        "summary": {
            "total_tools": len(execution),
            "succeeded_tools": succeeded,
            "failed_tools": failed,
        },
        "request_id": rid,
    }


@app.get("/readyz")
def readyz():
    model_ready = tokenizer is not None and model is not None
    cb = _circuit_breaker_snapshot()
    maintenance = _maintenance_snapshot()
    maintenance_active = _maintenance_is_active()
    ready = model_ready and cb["state"] != "open"
    return {
        "status": "ready" if ready else "not_ready",
        "model_loaded": model_ready,
        "circuit_breaker": cb,
        "maintenance": {
            "active": maintenance_active,
            "reason": maintenance.get("reason"),
            "read_only": bool(maintenance.get("read_only", False)),
            "expires_at": maintenance.get("expires_at"),
        },
        "cache": _cache_backend_name(),
        "persistence": _state_backend_name(),
        "request_id": REQUEST_ID.get(),
    }


@app.get("/model")
def model_info():
    return MODEL_META


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_require_chat_invoke)])
def chat(payload: ChatRequest):
    rid = REQUEST_ID.get()
    METRICS["chat_requests"] += 1

    key = _cache_key(payload)
    cached_response = _get_cached_chat_response(key)
    if cached_response is not None:
        METRICS["chat_cache_hits"] += 1
        log_json(
            LOGGER,
            {
                "event": "chat_cache_hit",
                "request_id": rid,
                "model": MODEL_NAME,
            },
        )
        return cached_response

    METRICS["chat_cache_misses"] += 1

    user_prompt = payload.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    slot_ok, slot_state = _chat_try_acquire_slot()
    if not slot_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "service_unavailable",
                "message": "chat backpressure: concurrency limit reached",
                "details": {
                    "reason": "backpressure",
                    "active_requests": slot_state["active_requests"],
                    "max_concurrent_requests": slot_state["max_concurrent_requests"],
                },
            },
        )

    try:
        allowed, cb_state = _circuit_breaker_try_acquire()
        if not allowed:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "service_unavailable",
                    "message": "inference temporarily unavailable (circuit breaker open)",
                    "details": {
                        "circuit_state": cb_state.get("state"),
                        "retry_after_seconds": cb_state.get("retry_after_seconds", 1),
                    },
                },
            )

        try:
            context_window = MODEL_META["context_window"]
            model_type = MODEL_META["model_type"]
            masking = MODEL_META["attention_masking"]
            heads = MODEL_META["attention_heads"]
            hidden = MODEL_META["hidden_size"]
            deadline = _chat_timeout_deadline()
            _enforce_chat_timeout(deadline, stage="pre_generation")

            # Run generation
            start = time.perf_counter()
            response_text, prompt_tokens, completion_tokens = _generate(
                prompt=user_prompt,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )
            _enforce_chat_timeout(deadline, stage="post_generation")

            refined = False
            original_response = None
            critique_text = None
            steps_used = 0

            # Optional self-refinement loop
            if payload.mode == "refine":
                refined = True
                original_response = response_text

                current_answer = response_text
                for _ in range(payload.refine_steps):
                    steps_used += 1
                    _enforce_chat_timeout(deadline, stage="pre_critique")

                    critique_prompt = _build_critique_prompt(user_prompt, current_answer)
                    critique_text, _, _ = _generate(
                        prompt=critique_prompt,
                        max_new_tokens=min(256, payload.max_new_tokens),
                        temperature=payload.critique_temperature,
                        top_p=payload.top_p,
                    )
                    _enforce_chat_timeout(deadline, stage="post_critique")
                    _enforce_chat_timeout(deadline, stage="pre_refine")

                    refine_prompt = _build_refine_prompt(user_prompt, current_answer, critique_text)
                    improved_answer, _, _ = _generate(
                        prompt=refine_prompt,
                        max_new_tokens=payload.max_new_tokens,
                        temperature=payload.temperature,
                        top_p=payload.top_p,
                    )
                    _enforce_chat_timeout(deadline, stage="post_refine")

                    current_answer = improved_answer

                # Final answer is the refined one
                response_text = current_answer

                # Update token counts for final prompt/answer visibility
                # Keep the original prompt_tokens as the user prompt tokens, but recompute completion tokens from final answer length
                # For this stage, we track totals at the API boundary rather than exact per-pass accounting.
                completion_tokens = _count_tokens(response_text)
        except Exception as exc:
            count_failure, reason = _is_circuit_breaker_failure(exc)
            if count_failure:
                _circuit_breaker_record_failure(reason=reason)
            else:
                _circuit_breaker_release_trial()
            raise

        _circuit_breaker_record_success()

        latency_ms = int((time.perf_counter() - start) * 1000)

        total_tokens = prompt_tokens + completion_tokens
        context_used_pct = round((total_tokens / context_window) * 100.0, 4)
        est_ops = _estimated_attention_ops(total_tokens)

        ratio = round((completion_tokens / max(prompt_tokens, 1)), 4)

        log_json(
            LOGGER,
            {
                "event": "chat",
                "request_id": rid,
                "model": MODEL_NAME,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "output_to_input_ratio": ratio,
                "context_window": context_window,
                "context_used_pct": context_used_pct,
                "model_type": model_type,
                "attention_masking": masking,
                "attention_heads": heads,
                "hidden_size": hidden,
                "estimated_attention_ops": est_ops,
                "mode": payload.mode,
                "refine_steps_used": steps_used,
                "temperature": payload.temperature,
                "top_p": payload.top_p,
                "max_new_tokens": payload.max_new_tokens,
            },
        )

        result = ChatResponse(
            response=response_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=MODEL_NAME,
            request_id=rid,
            context_window=context_window,
            context_used_pct=context_used_pct,
            model_type=model_type,
            attention_masking=masking,
            attention_heads=heads,
            hidden_size=hidden,
            estimated_attention_ops=est_ops,
            total_tokens=total_tokens,
            output_to_input_ratio=ratio,
            refined=refined,
            original_response=original_response,
            critique=critique_text,
            refine_steps_used=steps_used,
            cache_hit=False,
        )
        _put_cached_chat_response(key, result)
        return result
    finally:
        _chat_release_slot()


@app.post("/evaluate", response_model=EvaluateResponse, dependencies=[Depends(_require_evaluate_invoke)])
def evaluate(payload: EvaluateRequest):
    rid = REQUEST_ID.get()

    prompt = payload.prompt.strip()
    answer = payload.response.strip()

    if not prompt or not answer:
        raise HTTPException(status_code=400, detail="prompt and response must be non-empty")

    start = time.perf_counter()

    eval_prompt = _build_eval_prompt(prompt, answer, [c for c in payload.criteria])

    eval_text, _, _ = _generate(
        prompt=eval_prompt,
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    import re

    pattern = re.compile(
        r"criterion\s*:\s*(.+?)\s*\n"
        r"score\s*:\s*(\d+)\s*\n"
        r"rationale\s*:\s*(.+?)(?=\ncriterion:|\Z)",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(eval_text)

    scores: List[CriterionScore] = []

    for match in matches:
        try:
            criterion = match[0].strip()
            score = int(match[1].strip())
            rationale = match[2].strip()

            scores.append(
                CriterionScore(
                    criterion=criterion,
                    score=score,
                    rationale=rationale
                )
            )
        except Exception:
            continue

    if not scores:
        scores = [
            CriterionScore(
                criterion="overall",
                score=5,
                rationale=f"Failed to parse evaluator output. Raw output: {eval_text[:500]}"
            )
        ]

    log_json(
        LOGGER,
        {
            "event": "evaluate",
            "request_id": rid,
            "model": MODEL_NAME,
            "latency_ms": latency_ms,
            "criteria": payload.criteria,
            "scores": [s.model_dump() for s in scores],
        },
    )

    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    EVAL_RUNS[run_id] = {
        "run_id": run_id,
        "prompt": prompt,
        "response": answer,
        "criteria": list(payload.criteria),
        "scores": [s.model_dump() for s in scores],
        "model": MODEL_NAME,
        "latency_ms": latency_ms,
        "created_at": _now_ts(),
        "deleted_at": None,
    }
    _persist_eval_run_record(EVAL_RUNS[run_id])

    return EvaluateResponse(
        request_id=rid,
        model=MODEL_NAME,
        scores=scores,
        latency_ms=latency_ms,
        run_id=run_id,
    )


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(_require_chat_invoke)])
def v1_chat(payload: ChatRequest):
    return chat(payload)


@app.post("/v1/evaluate", response_model=EvaluateResponse, dependencies=[Depends(_require_evaluate_invoke)])
def v1_evaluate(payload: EvaluateRequest):
    return evaluate(payload)


@app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=[Depends(_require_embeddings_invoke)])
def v1_embeddings(payload: EmbeddingsRequest):
    rid = REQUEST_ID.get()
    inputs = payload.input if isinstance(payload.input, list) else [payload.input]
    data = []
    for idx, text in enumerate(inputs):
        token_count = max(1, len(text.split()))
        data.append({"index": idx, "embedding": [0.0] * min(8, token_count)})

    return EmbeddingsResponse(
        data=data,
        model=payload.model or "stub-embedding-model",
        usage={"input_tokens": sum(max(1, len(t.split())) for t in inputs), "total_tokens": sum(max(1, len(t.split())) for t in inputs)},
        request_id=rid,
    )



def _build_rag_contract_response(payload: RagQueryRequest) -> RagContractResponse:
    rid = REQUEST_ID.get()
    dataset = DATASETS.get(payload.dataset_id)
    if dataset is not None and dataset.get("deleted_at") is not None:
        dataset = None
    if dataset is None:
        dataset = _sqlite_dataset_row(payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset_id not found"})

    index_obj = RAG_INDEXES.get(payload.dataset_id)
    if index_obj is None or index_obj.get("deleted_at") is not None:
        index_obj = _sqlite_rag_index(payload.dataset_id)
    if index_obj is None:
        _upsert_rag_index(dataset_id=payload.dataset_id, records=dataset.get("records", []), now=_now_ts())
        index_obj = RAG_INDEXES.get(payload.dataset_id, {})
    hits = _search_rag_chunks(index_obj, payload.query, payload.top_k)
    if not hits:
        hits = [
            {
                "chunk_id": f"{payload.dataset_id}-chunk-1",
                "dataset_id": payload.dataset_id,
                "score": 0.0,
                "text": "No indexed chunks available yet.",
            }
        ]

    return RagContractResponse(
        answer="RAG retrieval response generated from indexed chunks.",
        citations=[
            {
                "doc_id": dataset["dataset_id"],
                "chunk_id": str(item.get("chunk_id", "")),
                "score": float(item.get("score", 0.0)),
                "text": str(item.get("text", "")),
            }
            for item in hits
        ],
        retrieval={"dataset_id": payload.dataset_id, "top_k": payload.top_k, "latency_ms": 8},
        generation={"model": MODEL_NAME, "latency_ms": 16},
        request_id=rid,
    )


@app.post("/v1/rag", response_model=RagContractResponse, dependencies=[Depends(_require_rag_query)])
def v1_rag(payload: RagQueryRequest):
    return _build_rag_contract_response(payload)

@app.post("/v1/rag/query", response_model=RagQueryResponse, dependencies=[Depends(_require_rag_query)])
def v1_rag_query(payload: RagQueryRequest):
    rid = REQUEST_ID.get()
    dataset = DATASETS.get(payload.dataset_id)
    if dataset is not None and dataset.get("deleted_at") is not None:
        dataset = None
    if dataset is None:
        dataset = _sqlite_dataset_row(payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset_id not found"})

    index_obj = RAG_INDEXES.get(payload.dataset_id)
    if index_obj is None or index_obj.get("deleted_at") is not None:
        index_obj = _sqlite_rag_index(payload.dataset_id)
    if index_obj is None:
        _upsert_rag_index(dataset_id=payload.dataset_id, records=dataset.get("records", []), now=_now_ts())
        index_obj = RAG_INDEXES.get(payload.dataset_id, {})
    hits = _search_rag_chunks(index_obj, payload.query, payload.top_k)
    if not hits:
        hits = [
            {
                "chunk_id": f"{payload.dataset_id}-chunk-1",
                "dataset_id": payload.dataset_id,
                "score": 0.0,
                "text": "No indexed chunks available yet.",
            }
        ]

    return RagQueryResponse(
        response="RAG retrieval response generated from indexed chunks.",
        retrieved_chunks=hits,
        request_id=rid,
    )


@app.post("/v1/datasets", response_model=DatasetCreateResponse, status_code=201, dependencies=[Depends(_require_datasets_write)])
def v1_create_dataset(payload: DatasetCreateRequest):
    rid = REQUEST_ID.get()
    now = _now_ts()
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    DATASETS[dataset_id] = {
        "dataset_id": dataset_id,
        "name": payload.name,
        "type": payload.type,
        "status": "ready",
        "records": payload.records,
        "record_count": len(payload.records),
        "metadata": payload.metadata,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    _persist_dataset_record(DATASETS[dataset_id])
    _upsert_rag_index(dataset_id=dataset_id, records=payload.records, now=now)
    return DatasetCreateResponse(
        dataset_id=dataset_id,
        name=payload.name,
        type=payload.type,
        status="ready",
        records_count=len(payload.records),
        created_at=now,
        request_id=rid,
    )




@app.post("/v1/datasets/upload", status_code=201, dependencies=[Depends(_require_datasets_write)])
async def v1_upload_dataset(
    name: str = Form(...),
    type: str = Form("eval_set"),
    file: UploadFile = File(...),
    metadata: str | None = Form(default=None),
):
    rid = REQUEST_ID.get()
    now = _now_ts()

    if type not in {"rag_corpus", "eval_set"}:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": "type must be rag_corpus or eval_set"},
        )

    raw = await file.read()
    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail={"code": "payload_too_large", "message": "dataset upload exceeds 100MB"},
        )

    metadata_obj: Dict[str, Any] = {}
    if metadata:
        try:
            loaded_meta = json.loads(metadata)
            metadata_obj = loaded_meta if isinstance(loaded_meta, dict) else {}
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_request", "message": "metadata must be valid JSON object"},
            )

    try:
        parsed = _parse_dataset_upload(file.filename or "", raw)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "file must be UTF-8"})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "invalid jsonl payload"})

    accepted, errors = _validate_records(parsed)

    if not accepted:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": "All records failed validation",
                "details": {"rejected_records": len(errors), "accepted_records": 0},
            },
        )

    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    DATASETS[dataset_id] = {
        "dataset_id": dataset_id,
        "name": name,
        "type": type,
        "status": "ready",
        "records": accepted,
        "record_count": len(accepted),
        "metadata": metadata_obj,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    _persist_dataset_record(DATASETS[dataset_id])
    _upsert_rag_index(dataset_id=dataset_id, records=accepted, now=now)

    if errors:
        return JSONResponse(
            status_code=202,
            content={
                "run_id": f"run_{uuid.uuid4().hex[:12]}",
                "status": "accepted_with_record_errors",
                "summary": {
                    "total_records": len(parsed),
                    "accepted_records": len(accepted),
                    "rejected_records": len(errors),
                },
                "record_errors": errors,
                "dataset_id": dataset_id,
                "request_id": rid,
            },
        )

    return DatasetCreateResponse(
        dataset_id=dataset_id,
        name=name,
        type=type,
        status="processing",
        records_count=len(accepted),
        created_at=now,
        request_id=rid,
    )

@app.get("/v1/datasets", response_model=DatasetListResponse, dependencies=[Depends(_require_datasets_read)])
def v1_list_datasets(
    type: str | None = None,
    status: str | None = None,
    limit: int = 20,
    include_deleted: bool = False,
):
    rid = REQUEST_ID.get()
    clamped_limit = max(1, min(limit, 100))
    sqlite_rows = _sqlite_dataset_rows(
        dataset_type=type,
        status=status,
        limit=clamped_limit,
        include_deleted=include_deleted,
    )
    if sqlite_rows is not None:
        items = [
            DatasetListItem(
                dataset_id=str(ds["dataset_id"]),
                name=str(ds["name"]),
                type=str(ds["type"]),
                status=str(ds["status"]),
                record_count=int(ds["record_count"]),
                created_at=int(ds["created_at"]),
            )
            for ds in sqlite_rows
        ]
        return DatasetListResponse(data=items, next_cursor=None, request_id=rid)

    items: List[DatasetListItem] = []
    for ds in DATASETS.values():
        if (not include_deleted) and ds.get("deleted_at") is not None:
            continue
        if type is not None and ds["type"] != type:
            continue
        if status is not None and ds["status"] != status:
            continue
        items.append(
            DatasetListItem(
                dataset_id=ds["dataset_id"],
                name=ds["name"],
                type=ds["type"],
                status=ds["status"],
                record_count=ds["record_count"],
                created_at=ds["created_at"],
            )
        )

    items = sorted(items, key=lambda x: x.created_at, reverse=True)[:clamped_limit]
    return DatasetListResponse(data=items, next_cursor=None, request_id=rid)


@app.get("/v1/datasets/{dataset_id}", response_model=DatasetGetResponse, dependencies=[Depends(_require_datasets_read)])
def v1_get_dataset(dataset_id: str, include_deleted: bool = False):
    rid = REQUEST_ID.get()
    ds = _sqlite_dataset_row(dataset_id, include_deleted=include_deleted)
    if ds is None:
        ds = DATASETS.get(dataset_id)
    if ds is not None and ds.get("deleted_at") is not None and not include_deleted:
        ds = None
    if ds is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset not found"})

    return DatasetGetResponse(
        dataset_id=ds["dataset_id"],
        name=ds["name"],
        type=ds["type"],
        status=ds["status"],
        record_count=ds["record_count"],
        error=ds["error"],
        created_at=ds["created_at"],
        updated_at=ds["updated_at"],
        request_id=rid,
    )


@app.delete("/v1/datasets/{dataset_id}", dependencies=[Depends(_require_datasets_write)])
def v1_delete_dataset(dataset_id: str):
    rid = REQUEST_ID.get()
    deleted = _delete_dataset_record(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset not found"})
    _record_audit_event(action="dataset.delete", resource_type="dataset", resource_id=dataset_id)
    return {"dataset_id": dataset_id, "deleted": True, "request_id": rid}


@app.post("/v1/datasets/{dataset_id}/restore", dependencies=[Depends(_require_datasets_write)])
def v1_restore_dataset(dataset_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_dataset_record(dataset_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset not found"})
    _record_audit_event(action="dataset.restore", resource_type="dataset", resource_id=dataset_id)
    return {"dataset_id": dataset_id, "restored": True, "request_id": rid}


@app.delete("/v1/datasets/{dataset_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_dataset(dataset_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_dataset_record(dataset_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset not found"})
    _record_audit_event(action="dataset.purge", resource_type="dataset", resource_id=dataset_id)
    return {"dataset_id": dataset_id, "purged": True, "request_id": rid}


@app.post("/v1/batch-evals", response_model=BatchEvalCreateResponse, status_code=202, dependencies=[Depends(_require_batch_write)])
def v1_create_batch_eval(payload: BatchEvalCreateRequest):
    rid = REQUEST_ID.get()
    dataset = DATASETS.get(payload.dataset_id)
    if dataset is not None and dataset.get("deleted_at") is not None:
        dataset = None
    if dataset is None:
        dataset = _sqlite_dataset_row(payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset_id not found"})

    run_id = f"be_{uuid.uuid4().hex[:8]}"
    now = _now_ts()
    progress = _build_batch_progress(total=len(dataset["records"]))
    criteria = [c for c in payload.criteria]

    global BATCH_QUEUE_SEQ
    with BATCH_EVAL_LOCK:
        BATCH_QUEUE_SEQ += 1
        BATCH_EVAL_RUNS[run_id] = {
            "run_id": run_id,
            "batch_eval_id": run_id,
            "dataset_id": payload.dataset_id,
            "status": "queued",
            "progress": progress,
            "criteria": criteria,
            "model": payload.model or MODEL_NAME,
            "concurrency": payload.concurrency,
            "summary": {"mean_scores": {}, "total_items": len(dataset["records"]), "failed_items": 0},
            "record_scores": [],
            "failures": [],
            "started_at": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "deleted_at": None,
            "events": [],
            "cancel_requested": False,
            "retries": [],
            "retry_stats": {"attempted_records": 0, "total_retries": 0, "exhausted_records": 0},
            "worker_started": False,
            "queue_seq": BATCH_QUEUE_SEQ,
        }
        _append_batch_event(
            BATCH_EVAL_RUNS[run_id],
            event_type="queued",
            details={"status": "queued", "total": len(dataset["records"])},
        )
        run_snapshot = dict(BATCH_EVAL_RUNS[run_id])
    _persist_batch_eval_run_record(run_snapshot)
    _dispatch_batch_workers()
    return BatchEvalCreateResponse(batch_eval_id=run_id, run_id=run_id, status="queued", created_at=now, request_id=rid)

@app.get("/v1/evals/{run_id}", response_model=EvalRunSummary, dependencies=[Depends(_require_evals_read)])
def v1_get_eval_run(run_id: str, include_deleted: bool = False):
    run = _sqlite_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "eval run not found"})

    return EvalRunSummary(
        run_id=run["run_id"],
        prompt=run["prompt"],
        response=run["response"],
        criteria=run["criteria"],
        scores=[CriterionScore.model_validate(s) for s in run["scores"]],
        model=run["model"],
        latency_ms=run["latency_ms"],
        created_at=run["created_at"],
    )


@app.delete("/v1/evals/{run_id}", dependencies=[Depends(_require_evals_write)])
def v1_delete_eval_run(run_id: str):
    rid = REQUEST_ID.get()
    deleted = _delete_eval_run_record(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "eval run not found"})
    _record_audit_event(action="eval.delete", resource_type="eval_run", resource_id=run_id)
    return {"run_id": run_id, "deleted": True, "request_id": rid}


@app.post("/v1/evals/{run_id}/restore", dependencies=[Depends(_require_evals_write)])
def v1_restore_eval_run(run_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_eval_run_record(run_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "eval run not found"})
    _record_audit_event(action="eval.restore", resource_type="eval_run", resource_id=run_id)
    return {"run_id": run_id, "restored": True, "request_id": rid}


@app.delete("/v1/evals/{run_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_eval_run(run_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_eval_run_record(run_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "eval run not found"})
    _record_audit_event(action="eval.purge", resource_type="eval_run", resource_id=run_id)
    return {"run_id": run_id, "purged": True, "request_id": rid}


@app.get("/v1/batch-evals/queue", dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval_queue(include_deleted: bool = False):
    rid = REQUEST_ID.get()
    with BATCH_EVAL_LOCK:
        runs = [dict(run) for run in BATCH_EVAL_RUNS.values() if isinstance(run, dict)]

    if not include_deleted:
        runs = [run for run in runs if run.get("deleted_at") is None]

    running = sorted(
        [run for run in runs if str(run.get("status")) == "running"],
        key=lambda x: (int(x.get("started_at", 0) or 0), int(x.get("created_at", 0) or 0), str(x.get("run_id", ""))),
    )
    queued = sorted(
        [run for run in runs if str(run.get("status")) == "queued"],
        key=lambda x: (int(x.get("queue_seq", 0) or 0), int(x.get("created_at", 0) or 0), str(x.get("run_id", ""))),
    )

    queued_items: List[Dict[str, Any]] = []
    for idx, run in enumerate(queued, start=1):
        queued_items.append(
            {
                "run_id": run.get("run_id"),
                "batch_eval_id": run.get("batch_eval_id", run.get("run_id")),
                "dataset_id": run.get("dataset_id"),
                "status": "queued",
                "queue_position": idx,
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
            }
        )

    running_items: List[Dict[str, Any]] = []
    for run in running:
        running_items.append(
            {
                "run_id": run.get("run_id"),
                "batch_eval_id": run.get("batch_eval_id", run.get("run_id")),
                "dataset_id": run.get("dataset_id"),
                "status": "running",
                "queue_position": None,
                "created_at": run.get("created_at"),
                "started_at": run.get("started_at"),
                "updated_at": run.get("updated_at"),
            }
        )

    return {
        "max_concurrent_runs": max(1, int(BATCH_EVAL_MAX_CONCURRENT_RUNS)),
        "running_count": len(running_items),
        "queued_count": len(queued_items),
        "running": running_items,
        "queued": queued_items,
        "request_id": rid,
    }


@app.get("/v1/batch-evals/{run_id}", response_model=BatchEvalStatusResponse, dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval(run_id: str, include_deleted: bool = False):
    rid = REQUEST_ID.get()
    run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    return BatchEvalStatusResponse(
        batch_eval_id=run.get("batch_eval_id", run["run_id"]),
        run_id=run["run_id"],
        dataset_id=run["dataset_id"],
        status=run["status"],
        progress=run["progress"],
        criteria=run["criteria"],
        started_at=run.get("started_at"),
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        request_id=rid,
    )


@app.delete("/v1/batch-evals/{run_id}", dependencies=[Depends(_require_batch_write)])
def v1_delete_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    deleted = _delete_batch_eval_run_record(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
    _record_audit_event(action="batch_eval.delete", resource_type="batch_eval_run", resource_id=run_id)
    return {"run_id": run_id, "deleted": True, "request_id": rid}


@app.post("/v1/batch-evals/{run_id}/cancel", dependencies=[Depends(_require_batch_write)])
def v1_cancel_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    run = _cancel_batch_eval_run_record(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
    _persist_batch_eval_run_record(run)
    _dispatch_batch_workers()
    _record_audit_event(
        action="batch_eval.cancel",
        resource_type="batch_eval_run",
        resource_id=run_id,
        details={"status": run.get("status"), "cancel_requested": bool(run.get("cancel_requested"))},
    )
    return {
        "run_id": run_id,
        "status": run.get("status"),
        "cancel_requested": bool(run.get("cancel_requested")),
        "request_id": rid,
    }


@app.post("/v1/batch-evals/{run_id}/restore", dependencies=[Depends(_require_batch_write)])
def v1_restore_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_batch_eval_run_record(run_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
    _record_audit_event(action="batch_eval.restore", resource_type="batch_eval_run", resource_id=run_id)
    return {"run_id": run_id, "restored": True, "request_id": rid}


@app.delete("/v1/batch-evals/{run_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_batch_eval_run_record(run_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
    _record_audit_event(action="batch_eval.purge", resource_type="batch_eval_run", resource_id=run_id)
    return {"run_id": run_id, "purged": True, "request_id": rid}


@app.get("/v1/rag/indexes/{index_id}", response_model=RagIndexStatusResponse, dependencies=[Depends(_require_rag_read)])
def v1_get_rag_index(index_id: str, include_deleted: bool = False):
    rid = REQUEST_ID.get()
    idx = _sqlite_rag_index(index_id, include_deleted=include_deleted)
    if idx is None:
        idx = RAG_INDEXES.get(index_id)
    if idx is not None and idx.get("deleted_at") is not None and not include_deleted:
        idx = None
    if idx is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})

    return RagIndexStatusResponse(
        index_id=idx["index_id"],
        dataset_id=idx["dataset_id"],
        status=idx["status"],
        chunk_count=idx["chunk_count"],
        updated_at=idx["updated_at"],
        request_id=rid,
    )


@app.get("/v1/rag/vector-backend", dependencies=[Depends(_require_rag_read)])
def v1_rag_vector_backend():
    payload = _rag_vector_backend_snapshot()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.delete("/v1/rag/indexes/{index_id}", dependencies=[Depends(_require_rag_write)])
def v1_delete_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    deleted = _delete_rag_index_record(index_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})
    _record_audit_event(action="rag_index.delete", resource_type="rag_index", resource_id=index_id)
    return {"index_id": index_id, "deleted": True, "request_id": rid}


@app.post("/v1/rag/indexes/{index_id}/restore", dependencies=[Depends(_require_rag_write)])
def v1_restore_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_rag_index_record(index_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})
    _record_audit_event(action="rag_index.restore", resource_type="rag_index", resource_id=index_id)
    return {"index_id": index_id, "restored": True, "request_id": rid}


@app.delete("/v1/rag/indexes/{index_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_rag_index_record(index_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})
    _record_audit_event(action="rag_index.purge", resource_type="rag_index", resource_id=index_id)
    return {"index_id": index_id, "purged": True, "request_id": rid}


@app.get("/v1/batch-evals/{run_id}/result", response_model=BatchEvalResultResponse, dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval_result(run_id: str, include_deleted: bool = False):
    rid = REQUEST_ID.get()
    run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    return BatchEvalResultResponse(
        batch_eval_id=run.get("batch_eval_id", run["run_id"]),
        status=run["status"],
        summary=run.get("summary", {"mean_scores": {}, "total_items": 0, "failed_items": 0}),
        completed_at=run.get("completed_at"),
        request_id=rid,
    )


@app.get("/v1/batch-evals/{run_id}/failures", response_model=BatchEvalFailuresResponse, dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval_failures(run_id: str, limit: int = 20, include_deleted: bool = False):
    rid = REQUEST_ID.get()
    run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    clamped_limit = max(1, min(limit, 100))
    data = [BatchEvalFailureItem.model_validate(item) for item in run.get("failures", [])[:clamped_limit]]
    return BatchEvalFailuresResponse(
        batch_eval_id=run.get("batch_eval_id", run["run_id"]),
        data=data,
        count=len(run.get("failures", [])),
        request_id=rid,
    )


@app.get("/v1/batch-evals/{run_id}/distribution", response_model=BatchEvalDistributionResponse, dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval_distribution(run_id: str, criterion: str = "overall", include_deleted: bool = False):
    rid = REQUEST_ID.get()
    run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    criteria = run.get("criteria", [])
    if criterion not in criteria:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": "criterion not present in batch run criteria",
                "details": {"criterion": criterion, "available": criteria},
            },
        )

    dist = _compute_distribution(run.get("record_scores", []), criterion)
    return BatchEvalDistributionResponse(
        batch_eval_id=run.get("batch_eval_id", run["run_id"]),
        criterion=criterion,
        buckets=dist["buckets"],
        summary=dist["summary"],
        request_id=rid,
    )


@app.get("/v1/batch-evals/{run_id}/artifacts/export", dependencies=[Depends(_require_batch_read)])
def v1_export_batch_eval_artifacts(
    run_id: str,
    format: str = "json",
    include_deleted: bool = False,
    include_records: bool = True,
    failures_limit: int = 200,
):
    rid = REQUEST_ID.get()
    run = _get_batch_eval_run_or_404(run_id, include_deleted=include_deleted)

    normalized_format = (format or "json").strip().lower()
    if normalized_format not in {"json", "jsonl", "csv"}:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": "format must be one of: json, jsonl, csv",
                "details": {"format": format},
            },
        )

    _record_audit_event(
        action="batch_eval.export",
        resource_type="batch_eval_run",
        resource_id=run_id,
        details={
            "format": normalized_format,
            "include_deleted": include_deleted,
            "include_records": include_records,
            "failures_limit": max(1, min(int(failures_limit), 5000)),
        },
    )

    base_name = f"batch_eval_{run_id}_artifacts"
    if normalized_format == "json":
        payload = _build_batch_artifact_export_payload(
            run=run,
            request_id=rid,
            include_records=include_records,
            failures_limit=failures_limit,
        )
        payload["filename"] = f"{base_name}.json"
        return payload

    if normalized_format == "csv":
        data = _render_batch_artifact_export_csv(run=run, include_records=include_records)
        headers = {"Content-Disposition": f'attachment; filename="{base_name}.csv"'}
        return PlainTextResponse(data, media_type="text/csv; charset=utf-8", headers=headers)

    data = _render_batch_artifact_export_jsonl(run=run, include_records=include_records)
    headers = {"Content-Disposition": f'attachment; filename="{base_name}.jsonl"'}
    return PlainTextResponse(data, media_type="application/x-ndjson; charset=utf-8", headers=headers)


@app.get("/v1/batch-evals/{run_id}/events", dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval_events(
    run_id: str,
    limit: int = 100,
    event_type: str | None = None,
    include_deleted: bool = False,
):
    rid = REQUEST_ID.get()
    run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if run is None:
        run = BATCH_EVAL_RUNS.get(run_id)
    if run is not None and run.get("deleted_at") is not None and not include_deleted:
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    events = run.get("events", [])
    if not isinstance(events, list):
        events = []
    if event_type:
        events = [e for e in events if isinstance(e, dict) and e.get("event_type") == event_type]

    clamped_limit = max(1, min(limit, 1000))
    data = list(reversed(events))[:clamped_limit]
    return {
        "batch_eval_id": run.get("batch_eval_id", run["run_id"]),
        "run_id": run["run_id"],
        "data": data,
        "count": len(data),
        "request_id": rid,
    }


@app.get("/v1/batch-evals/{run_id}/retries", dependencies=[Depends(_require_batch_read)])
def v1_get_batch_eval_retries(
    run_id: str,
    limit: int = 100,
    include_deleted: bool = False,
):
    rid = REQUEST_ID.get()
    run = _get_batch_eval_run_or_404(run_id, include_deleted=include_deleted)
    retries = run.get("retries", [])
    if not isinstance(retries, list):
        retries = []
    clamped_limit = max(1, min(int(limit), 1000))
    data = list(reversed(retries))[:clamped_limit]
    return {
        "batch_eval_id": run.get("batch_eval_id", run["run_id"]),
        "run_id": run["run_id"],
        "retry_stats": run.get("retry_stats", {"attempted_records": 0, "total_retries": 0, "exhausted_records": 0}),
        "count": len(data),
        "data": data,
        "request_id": rid,
    }


@app.get("/v1/batch-evals/{run_id}/events/stream", dependencies=[Depends(_require_batch_read)])
async def v1_stream_batch_eval_events(
    run_id: str,
    event_type: str | None = None,
    since_ts: int | None = None,
    include_deleted: bool = False,
    poll_interval_ms: int = 500,
    max_seconds: int = 30,
):
    rid = REQUEST_ID.get()
    initial_run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
    if initial_run is None:
        initial_run = BATCH_EVAL_RUNS.get(run_id)
    if initial_run is not None and initial_run.get("deleted_at") is not None and not include_deleted:
        initial_run = None
    if initial_run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    sleep_s = max(0.05, min(float(poll_interval_ms) / 1000.0, 10.0))
    max_duration_s = max(1, min(int(max_seconds), 600))
    cursor = int(since_ts or 0)

    async def _event_gen():
        cursor_ts = cursor
        started = time.monotonic()
        yield _to_sse_event("meta", {"run_id": run_id, "request_id": rid, "poll_interval_ms": int(sleep_s * 1000)})

        while True:
            run = _sqlite_batch_eval_run(run_id, include_deleted=include_deleted)
            if run is None:
                run = BATCH_EVAL_RUNS.get(run_id)
            if run is not None and run.get("deleted_at") is not None and not include_deleted:
                run = None

            if run is None:
                yield _to_sse_event("error", {"code": "not_found", "message": "batch eval run not found", "run_id": run_id})
                break

            new_events = _slice_batch_events(run, since_ts=cursor_ts, event_type=event_type, limit=1000, ascending=True)
            for item in new_events:
                event_ts = int(item.get("ts", 0) or 0)
                cursor_ts = max(cursor_ts, event_ts)
                yield _to_sse_event("batch_event", item)

            terminal = run.get("status") in {"completed", "failed"}
            if terminal:
                yield _to_sse_event("done", {"run_id": run_id, "status": run.get("status"), "last_ts": cursor_ts})
                break

            if (time.monotonic() - started) >= max_duration_s:
                yield _to_sse_event("timeout", {"run_id": run_id, "last_ts": cursor_ts, "max_seconds": max_duration_s})
                break

            await asyncio.sleep(sleep_s)

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


@app.get("/v1/metrics", dependencies=[Depends(_require_metrics_read)])
def v1_metrics(format: str = "json"):
    if format in {"text", "prometheus", "openmetrics"}:
        return PlainTextResponse(_render_metrics_text(), media_type="text/plain; version=0.0.4")
    counts = _state_counts()
    cb = _circuit_breaker_snapshot()
    chat_runtime = _chat_concurrency_snapshot()
    maintenance = _maintenance_snapshot()
    maintenance_active = _maintenance_is_active()
    slo = _slo_snapshot()
    with SLO_LOCK:
        incident_count = len(SLO_INCIDENTS)
        open_incident_count = sum(1 for x in SLO_INCIDENTS if str(x.get("status", "")) == "open")

    return {
        "request_id": REQUEST_ID.get(),
        "model": MODEL_NAME,
        "chat": {
            "requests": METRICS["chat_requests"],
            "active_requests": chat_runtime["active_requests"],
            "max_concurrent_requests": chat_runtime["max_concurrent_requests"],
            "backpressure_rejections": int(METRICS.get("chat_backpressure_rejections", 0)),
            "timeouts": int(METRICS.get("chat_timeouts", 0)),
            "maintenance_rejections": int(METRICS.get("maintenance_rejections", 0)),
        },
        "datasets": {"count": counts["datasets"]},
        "eval_runs": {"count": counts["eval_runs"]},
        "batch_eval_runs": {"count": counts["batch_eval_runs"]},
        "runbook_runs": {"count": counts["runbook_runs"]},
        "runbook_templates": {"count": counts["runbook_templates"]},
        "audit_logs": {"count": counts["audit_logs"]},
        "cache": _cache_stats(),
        "rate_limit": {
            "requests_per_minute_per_key": RATE_LIMIT_PER_MINUTE,
            "active_buckets": len(RATE_LIMIT_BUCKETS),
        },
        "persistence": {
            "backend": _state_backend_name(),
        },
        "tracing": _tracing_backend_snapshot(),
        "circuit_breaker": cb,
        "maintenance": {
            "active": maintenance_active,
            "reason": maintenance.get("reason"),
            "read_only": bool(maintenance.get("read_only", False)),
            "enabled_at": maintenance.get("enabled_at"),
            "expires_at": maintenance.get("expires_at"),
        },
        "slo": slo,
        "slo_incidents": {
            "count": incident_count,
            "open_count": open_incident_count,
        },
        "retention": {
            "enabled": RETENTION_SWEEP_ENABLED,
            "retention_seconds": max(SOFT_DELETE_RETENTION_SECONDS, 0),
            "sweep_interval_seconds": max(RETENTION_SWEEP_INTERVAL_SECONDS, 5),
            "last_run_ts": RETENTION_STATS.get("last_run_ts"),
            "last_purged_total": RETENTION_STATS.get("last_purged_total", 0),
            "last_error": RETENTION_STATS.get("last_error"),
            "history_limit": max(1, RETENTION_HISTORY_LIMIT),
            "recent_runs": list(RETENTION_HISTORY),
        },
    }


@app.get("/v1/tracing/status", dependencies=[Depends(_require_metrics_read)])
def v1_tracing_status():
    payload = _tracing_backend_snapshot()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/metrics/dashboard", dependencies=[Depends(_require_metrics_read)])
def metrics_dashboard(window: str = "15m"):
    cache = _cache_stats()
    counts = _state_counts()
    queue_depth = _batch_queue_depth()
    total_chat = METRICS["chat_requests"]
    rejected = sum(max(v["count"] - RATE_LIMIT_PER_MINUTE, 0) for v in RATE_LIMIT_BUCKETS.values())
    cb = _circuit_breaker_snapshot()
    chat_runtime = _chat_concurrency_snapshot()
    maintenance = _maintenance_snapshot()
    maintenance_active = _maintenance_is_active()
    slo = _slo_snapshot()
    with SLO_LOCK:
        incident_count = len(SLO_INCIDENTS)
        open_incident_count = sum(1 for x in SLO_INCIDENTS if str(x.get("status", "")) == "open")

    degraded = tokenizer is None or model is None or cb["state"] == "open" or maintenance_active
    rag_backend = _rag_vector_backend_snapshot()

    return {
        "request_id": REQUEST_ID.get(),
        "window": window,
        "service": {
            "status": "degraded" if degraded else "ok",
            "model": MODEL_NAME,
            "device": DEVICE,
        },
        "kpis": {
            "requests_per_minute_estimate": total_chat,
            "cache_hit_rate": cache["hit_rate"],
            "rate_limit_rejections": rejected,
            "datasets": counts["datasets"],
            "batch_runs": counts["batch_eval_runs"],
            "runbooks": counts["runbook_runs"],
            "runbook_templates": counts["runbook_templates"],
        },
        "rag_vector_backend": rag_backend,
        "tracing": _tracing_backend_snapshot(),
        "cache": cache,
        "latency": {
            "note": "Use x-latency-ms response header and external scrape for p50/p95/p99 trends.",
        },
        "errors": {
            "rate_limited": rejected,
            "retention_last_error": RETENTION_STATS.get("last_error"),
            "chat_backpressure_rejections": int(METRICS.get("chat_backpressure_rejections", 0)),
            "chat_timeouts": int(METRICS.get("chat_timeouts", 0)),
            "maintenance_rejections": int(METRICS.get("maintenance_rejections", 0)),
        },
        "circuit_breaker": cb,
        "maintenance": {
            "active": maintenance_active,
            "reason": maintenance.get("reason"),
            "read_only": bool(maintenance.get("read_only", False)),
            "expires_at": maintenance.get("expires_at"),
        },
        "slo": slo,
        "slo_incidents": {
            "count": incident_count,
            "open_count": open_incident_count,
        },
        "chat_runtime": {
            "active_requests": chat_runtime["active_requests"],
            "max_concurrent_requests": chat_runtime["max_concurrent_requests"],
            "timeout_ms": max(50, int(CHAT_REQUEST_TIMEOUT_MS)),
        },
        "queue_depth": {
            "pending": queue_depth["pending"],
            "running": queue_depth["running"],
        },
        "retention": {
            "enabled": RETENTION_SWEEP_ENABLED,
            "retention_seconds": max(SOFT_DELETE_RETENTION_SECONDS, 0),
            "last_run_ts": RETENTION_STATS.get("last_run_ts"),
            "last_purged_total": RETENTION_STATS.get("last_purged_total", 0),
            "recent_runs": list(RETENTION_HISTORY),
        },
    }


@app.post("/v1/admin/retention/sweep", dependencies=[Depends(_require_admin)])
def v1_admin_retention_sweep(retention_seconds: int | None = None):
    result = _purge_expired_soft_deleted_records(retention_seconds=retention_seconds)
    _record_retention_run(result=result, error=None, trigger="manual")
    _record_audit_event(
        action="retention.sweep",
        resource_type="retention",
        details={"retention_seconds": result.get("retention_seconds"), "purged_total": result.get("purged_total")},
    )
    result["request_id"] = REQUEST_ID.get()
    return result


@app.get("/v1/admin/retention/preview", dependencies=[Depends(_require_admin)])
def v1_admin_retention_preview(
    retention_seconds: int | None = None,
    include_ids: bool = True,
    candidate_ids_limit: int = 100,
):
    result = _preview_expired_soft_deleted_records(
        retention_seconds=retention_seconds,
        include_ids=include_ids,
        candidate_ids_limit=candidate_ids_limit,
    )
    _record_audit_event(
        action="retention.preview",
        resource_type="retention",
        details={
            "retention_seconds": result.get("retention_seconds"),
            "candidate_total": result.get("candidate_total"),
            "include_ids": include_ids,
            "candidate_ids_limit": candidate_ids_limit,
        },
    )
    result["request_id"] = REQUEST_ID.get()
    return result


@app.get("/v1/admin/rag/vector-backend", dependencies=[Depends(_require_admin)])
def v1_admin_rag_vector_backend():
    payload = _rag_vector_backend_snapshot()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/v1/admin/tracing/status", dependencies=[Depends(_require_admin)])
def v1_admin_tracing_status():
    payload = _tracing_backend_snapshot()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.post("/v1/admin/tracing/probe", dependencies=[Depends(_require_admin)])
def v1_admin_tracing_probe(name: str | None = None):
    rid = REQUEST_ID.get()
    label = (name or "manual").strip() or "manual"
    emitted = False
    if TRACING_ACTIVE and TRACER is not None:
        try:
            with TRACER.start_as_current_span(f"probe:{label}") as span:
                if span is not None:
                    span.set_attribute("request_id", rid)
                    span.set_attribute("probe_name", label)
            emitted = True
        except Exception as exc:
            return {
                "active": False,
                "emitted": False,
                "error": repr(exc),
                "request_id": rid,
            }
    payload = _tracing_backend_snapshot()
    payload.update({"probe_name": label, "emitted": emitted, "request_id": rid})
    return payload


@app.get("/v1/admin/runbooks", dependencies=[Depends(_require_admin)])
def v1_admin_runbooks(limit: int = 100, status: str | None = None):
    rid = REQUEST_ID.get()
    clamped_limit = max(1, min(int(limit), 500))
    normalized_status = (status or "").strip().lower()
    if normalized_status and normalized_status not in {"in_progress", "completed", "aborted"}:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "status must be in_progress, completed, or aborted"})
    rows = [dict(v) for v in RUNBOOK_RUNS.values() if isinstance(v, dict)]
    if normalized_status:
        rows = [x for x in rows if str(x.get("status", "")) == normalized_status]
    rows = sorted(rows, key=lambda x: int(x.get("created_at", 0) or 0), reverse=True)[:clamped_limit]
    return {"data": rows, "count": len(rows), "request_id": rid}


@app.get("/v1/admin/runbook-templates", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_templates(limit: int = 100):
    rid = REQUEST_ID.get()
    clamped_limit = max(1, min(int(limit), 500))
    rows = [dict(v) for v in RUNBOOK_TEMPLATES.values() if isinstance(v, dict)]
    rows = sorted(rows, key=lambda x: int(x.get("updated_at", 0) or 0), reverse=True)[:clamped_limit]
    return {"data": rows, "count": len(rows), "request_id": rid}


@app.get("/v1/admin/runbook-templates/{template_id}", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_template_get(template_id: str):
    rid = REQUEST_ID.get()
    tid = _validate_runbook_template_id(template_id)
    row = RUNBOOK_TEMPLATES.get(tid)
    if not isinstance(row, dict):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runbook template not found"})
    return {"data": dict(row), "request_id": rid}


@app.post("/v1/admin/runbook-templates/{template_id}", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_template_upsert(
    template_id: str,
    payload: Dict[str, Any] = Body(...),
    overwrite: bool = True,
):
    rid = REQUEST_ID.get()
    tid = _validate_runbook_template_id(template_id)
    if (tid in RUNBOOK_TEMPLATES) and (not overwrite):
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "runbook template already exists"})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "payload must be object"})
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "name must be non-empty"})
    environment = str(payload.get("environment", "local")).strip() or "local"
    steps = payload.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "steps must be non-empty array"})
    step_titles = [str(s or "").strip() for s in steps]
    if any(not x for x in step_titles):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "all steps must be non-empty strings"})
    now = _now_ts()
    row = {
        "template_id": tid,
        "name": name[:300],
        "environment": environment[:120],
        "steps": [x[:300] for x in step_titles],
        "updated_at": now,
    }
    RUNBOOK_TEMPLATES[tid] = row
    _save_state_to_disk()
    _record_audit_event(action="runbook_template.upsert", resource_type="runbook_template", resource_id=tid)
    return {"data": dict(row), "request_id": rid}


@app.delete("/v1/admin/runbook-templates/{template_id}", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_template_delete(template_id: str):
    rid = REQUEST_ID.get()
    tid = _validate_runbook_template_id(template_id)
    row = RUNBOOK_TEMPLATES.pop(tid, None)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runbook template not found"})
    _save_state_to_disk()
    _record_audit_event(action="runbook_template.delete", resource_type="runbook_template", resource_id=tid)
    return {"template_id": tid, "deleted": True, "request_id": rid}


@app.post("/v1/admin/runbooks", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_create(payload: Dict[str, Any] = Body(...)):
    rid = REQUEST_ID.get()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "payload must be object"})
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "name must be non-empty"})
    environment = str(payload.get("environment", "local")).strip() or "local"
    steps = _validate_runbook_steps(payload.get("steps", []))
    now = _now_ts()
    runbook_id = f"rb_{uuid.uuid4().hex[:10]}"
    row = {
        "runbook_id": runbook_id,
        "name": name[:300],
        "environment": environment[:120],
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "aborted_at": None,
        "steps": steps,
    }
    RUNBOOK_RUNS[runbook_id] = row
    _save_state_to_disk()
    _record_audit_event(action="runbook.create", resource_type="runbook", resource_id=runbook_id)
    return {"data": dict(row), "request_id": rid}


@app.post("/v1/admin/runbooks/from-template/{template_id}", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_create_from_template(template_id: str, name: str | None = None, environment: str | None = None):
    rid = REQUEST_ID.get()
    tid = _validate_runbook_template_id(template_id)
    template = RUNBOOK_TEMPLATES.get(tid)
    if not isinstance(template, dict):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runbook template not found"})
    step_titles = list(template.get("steps", []))
    steps = _validate_runbook_steps(step_titles)
    now = _now_ts()
    runbook_id = f"rb_{uuid.uuid4().hex[:10]}"
    row = {
        "runbook_id": runbook_id,
        "name": (str(name).strip() if name is not None else str(template.get("name", "runbook"))).strip()[:300],
        "environment": (str(environment).strip() if environment is not None else str(template.get("environment", "local"))).strip()[:120],
        "template_id": tid,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "aborted_at": None,
        "steps": steps,
    }
    RUNBOOK_RUNS[runbook_id] = row
    _save_state_to_disk()
    _record_audit_event(
        action="runbook.create_from_template",
        resource_type="runbook",
        resource_id=runbook_id,
        details={"template_id": tid},
    )
    return {"data": dict(row), "request_id": rid}


@app.get("/v1/admin/runbooks/{runbook_id}", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_get(runbook_id: str):
    rid = REQUEST_ID.get()
    row = _get_runbook_or_404(runbook_id)
    return {"data": dict(row), "request_id": rid}


@app.post("/v1/admin/runbooks/{runbook_id}/steps/{step_id}", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_step_update(
    runbook_id: str,
    step_id: str,
    payload: Dict[str, Any] = Body(...),
):
    rid = REQUEST_ID.get()
    row = _get_runbook_or_404(runbook_id)
    if str(row.get("status")) in {"completed", "aborted"}:
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "runbook is terminal"})
    status = str((payload or {}).get("status", "")).strip().lower()
    if status not in {"pending", "completed", "blocked"}:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "status must be pending, completed, or blocked"})
    note = (payload or {}).get("note")
    steps = row.get("steps")
    if not isinstance(steps, list):
        raise HTTPException(status_code=500, detail={"code": "internal_error", "message": "runbook steps malformed"})
    target = None
    for item in steps:
        if isinstance(item, dict) and str(item.get("step_id")) == str(step_id):
            target = item
            break
    if target is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "step not found"})
    now = _now_ts()
    target["status"] = status
    target["updated_at"] = now
    target["note"] = None if note is None else str(note)[:2000]
    target["completed_at"] = now if status == "completed" else None
    row["updated_at"] = now
    _save_state_to_disk()
    _record_audit_event(
        action="runbook.step_update",
        resource_type="runbook",
        resource_id=runbook_id,
        details={"step_id": str(step_id), "status": status},
    )
    return {"data": dict(row), "request_id": rid}


@app.post("/v1/admin/runbooks/{runbook_id}/complete", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_complete(runbook_id: str):
    rid = REQUEST_ID.get()
    row = _get_runbook_or_404(runbook_id)
    if str(row.get("status")) == "aborted":
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "runbook is aborted"})
    steps = row.get("steps", [])
    if not isinstance(steps, list):
        raise HTTPException(status_code=500, detail={"code": "internal_error", "message": "runbook steps malformed"})
    incomplete = [s for s in steps if isinstance(s, dict) and str(s.get("status")) != "completed"]
    if incomplete:
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "all steps must be completed first"})
    now = _now_ts()
    row["status"] = "completed"
    row["updated_at"] = now
    row["completed_at"] = now
    _save_state_to_disk()
    _record_audit_event(action="runbook.complete", resource_type="runbook", resource_id=runbook_id)
    return {"data": dict(row), "request_id": rid}


@app.post("/v1/admin/runbooks/{runbook_id}/abort", dependencies=[Depends(_require_admin)])
def v1_admin_runbook_abort(runbook_id: str, reason: str | None = None):
    rid = REQUEST_ID.get()
    row = _get_runbook_or_404(runbook_id)
    if str(row.get("status")) == "completed":
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "runbook already completed"})
    now = _now_ts()
    row["status"] = "aborted"
    row["updated_at"] = now
    row["aborted_at"] = now
    row["abort_reason"] = None if reason is None else str(reason)[:1000]
    _save_state_to_disk()
    _record_audit_event(
        action="runbook.abort",
        resource_type="runbook",
        resource_id=runbook_id,
        details={"reason": row.get("abort_reason")},
    )
    return {"data": dict(row), "request_id": rid}


@app.get("/v1/admin/maintenance", dependencies=[Depends(_require_admin)])
def v1_admin_maintenance_status():
    payload = _maintenance_snapshot()
    payload["active"] = _maintenance_is_active()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/v1/admin/slo/status", dependencies=[Depends(_require_admin)])
def v1_admin_slo_status(window_seconds: int | None = None):
    payload = _slo_snapshot(window_seconds=window_seconds)
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/v1/admin/slo/incidents", dependencies=[Depends(_require_admin)])
def v1_admin_slo_incidents(limit: int = 100, status: str | None = None):
    rid = REQUEST_ID.get()
    clamped_limit = max(1, min(int(limit), 500))
    normalized_status = (status or "").strip().lower()
    if normalized_status and normalized_status not in {"open", "resolved"}:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "status must be open or resolved"})

    with SLO_LOCK:
        rows = [dict(item) for item in SLO_INCIDENTS if isinstance(item, dict)]
    if normalized_status:
        rows = [r for r in rows if str(r.get("status", "")).strip().lower() == normalized_status]
    rows = sorted(rows, key=lambda x: int(x.get("opened_at", 0) or 0), reverse=True)[:clamped_limit]
    return {"data": rows, "count": len(rows), "request_id": rid}


@app.get("/v1/admin/slo/incidents/current", dependencies=[Depends(_require_admin)])
def v1_admin_slo_incident_current():
    rid = REQUEST_ID.get()
    with SLO_LOCK:
        current_id = SLO_STATE.get("current_incident_id")
        row = None
        if current_id:
            for item in reversed(SLO_INCIDENTS):
                if str(item.get("incident_id")) == str(current_id):
                    row = dict(item)
                    break
    return {"data": row, "request_id": rid}


@app.get("/v1/admin/slo/incidents/{incident_id}", dependencies=[Depends(_require_admin)])
def v1_admin_slo_incident_get(incident_id: str):
    rid = REQUEST_ID.get()
    row = _find_slo_incident(incident_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "slo incident not found"})
    return {"data": dict(row), "request_id": rid}


@app.post("/v1/admin/slo/incidents/{incident_id}/ack", dependencies=[Depends(_require_admin)])
def v1_admin_slo_incident_ack(incident_id: str, note: str | None = None):
    rid = REQUEST_ID.get()
    row = _find_slo_incident(incident_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "slo incident not found"})
    now = _now_ts()
    auth = AUTH_CONTEXT.get() or {}
    with SLO_LOCK:
        row["acknowledged_at"] = now
        row["acknowledged_by"] = str(auth.get("role", "unknown"))
        row["updated_at"] = now
        if note:
            notes = row.get("notes")
            if not isinstance(notes, list):
                notes = []
                row["notes"] = notes
            notes.append({"ts": now, "type": "ack_note", "text": str(note)[:2000], "actor_role": str(auth.get("role", "unknown"))})
            if len(notes) > 200:
                del notes[:-200]
    _save_state_to_disk()
    _record_audit_event(
        action="slo_incident.ack",
        resource_type="slo_incident",
        resource_id=incident_id,
        details={"has_note": bool(note)},
    )
    return {"incident_id": incident_id, "acknowledged": True, "request_id": rid}


@app.post("/v1/admin/slo/incidents/{incident_id}/notes", dependencies=[Depends(_require_admin)])
def v1_admin_slo_incident_add_note(incident_id: str, note: str = Body(..., embed=True)):
    rid = REQUEST_ID.get()
    row = _find_slo_incident(incident_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "slo incident not found"})
    text = str(note or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "note must be non-empty"})
    now = _now_ts()
    auth = AUTH_CONTEXT.get() or {}
    with SLO_LOCK:
        notes = row.get("notes")
        if not isinstance(notes, list):
            notes = []
            row["notes"] = notes
        notes.append({"ts": now, "type": "note", "text": text[:2000], "actor_role": str(auth.get("role", "unknown"))})
        if len(notes) > 200:
            del notes[:-200]
        row["updated_at"] = now
    _save_state_to_disk()
    _record_audit_event(
        action="slo_incident.note",
        resource_type="slo_incident",
        resource_id=incident_id,
        details={"note_length": len(text)},
    )
    return {"incident_id": incident_id, "noted": True, "request_id": rid}


@app.post("/v1/admin/maintenance/enable", dependencies=[Depends(_require_admin)])
def v1_admin_maintenance_enable(
    reason: str | None = None,
    duration_seconds: int | None = None,
    read_only: bool = False,
):
    if duration_seconds is not None and int(duration_seconds) <= 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": "duration_seconds must be positive when provided"},
        )
    payload = _maintenance_enable(reason=reason, duration_seconds=duration_seconds, read_only=read_only)
    _persist_runtime_profiles_if_needed()
    _record_audit_event(
        action="maintenance.enable",
        resource_type="maintenance",
        details={"reason": payload.get("reason"), "duration_seconds": duration_seconds, "read_only": bool(read_only)},
    )
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.post("/v1/admin/maintenance/disable", dependencies=[Depends(_require_admin)])
def v1_admin_maintenance_disable():
    payload = _maintenance_disable()
    _persist_runtime_profiles_if_needed()
    _record_audit_event(action="maintenance.disable", resource_type="maintenance")
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/v1/admin/circuit-breaker", dependencies=[Depends(_require_admin)])
def v1_admin_circuit_breaker_status():
    payload = _circuit_breaker_snapshot()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/v1/admin/runtime-config", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config():
    payload = _runtime_config_snapshot()
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.post("/v1/admin/runtime-config", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config_update(
    payload: Dict[str, Any] = Body(...),
    dry_run: bool = False,
):
    rid = REQUEST_ID.get()
    validated = _validate_runtime_config_patch(payload)
    before = _runtime_config_snapshot()
    after = dict(before)
    after.update(validated)
    if not dry_run:
        after = _apply_runtime_config_patch(validated)
        _record_audit_event(
            action="runtime_config.update",
            resource_type="runtime_config",
            details={"dry_run": False, "keys": sorted(validated.keys()), "before": before, "after": after},
        )
    else:
        _record_audit_event(
            action="runtime_config.update",
            resource_type="runtime_config",
            details={"dry_run": True, "keys": sorted(validated.keys()), "before": before, "after": after},
        )
    return {
        "dry_run": bool(dry_run),
        "applied": not bool(dry_run),
        "before": before,
        "after": after,
        "request_id": rid,
    }


@app.get("/v1/admin/runtime-config/profiles", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config_profiles(limit: int = 100):
    rid = REQUEST_ID.get()
    clamped_limit = max(1, min(int(limit), 500))
    rows = [
        {
            "profile_name": name,
            "updated_at": int(payload.get("updated_at", 0) or 0),
            "keys": sorted(list((payload.get("config") or {}).keys())) if isinstance(payload, dict) else [],
        }
        for name, payload in RUNTIME_CONFIG_PROFILES.items()
        if isinstance(payload, dict)
    ]
    rows = sorted(rows, key=lambda x: int(x.get("updated_at", 0) or 0), reverse=True)[:clamped_limit]
    return {"data": rows, "count": len(rows), "request_id": rid}


@app.get("/v1/admin/runtime-config/profiles/{profile_name}", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config_profile_get(profile_name: str):
    rid = REQUEST_ID.get()
    name = _validate_runtime_profile_name(profile_name)
    profile = RUNTIME_CONFIG_PROFILES.get(name)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runtime config profile not found"})
    out = dict(profile)
    out["request_id"] = rid
    return out


@app.post("/v1/admin/runtime-config/profiles/{profile_name}", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config_profile_upsert(
    profile_name: str,
    payload: Dict[str, Any] | None = Body(default=None),
    overwrite: bool = True,
):
    rid = REQUEST_ID.get()
    name = _validate_runtime_profile_name(profile_name)
    exists = name in RUNTIME_CONFIG_PROFILES
    if exists and not overwrite:
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "profile already exists"})

    patch = payload or {}
    validated_patch = _validate_runtime_config_patch(patch) if patch else {}
    base = _runtime_config_snapshot()
    config = dict(base)
    config.update(validated_patch)
    profile = {
        "profile_name": name,
        "config": config,
        "updated_at": _now_ts(),
    }
    RUNTIME_CONFIG_PROFILES[name] = profile
    _persist_runtime_profiles_if_needed()
    _record_audit_event(
        action="runtime_config_profile.upsert",
        resource_type="runtime_config_profile",
        resource_id=name,
        details={"overwrite": bool(overwrite), "patch_keys": sorted(validated_patch.keys())},
    )
    out = dict(profile)
    out["request_id"] = rid
    return out


@app.post("/v1/admin/runtime-config/profiles/{profile_name}/apply", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config_profile_apply(profile_name: str, dry_run: bool = False):
    rid = REQUEST_ID.get()
    name = _validate_runtime_profile_name(profile_name)
    profile = RUNTIME_CONFIG_PROFILES.get(name)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runtime config profile not found"})

    config = profile.get("config", {})
    validated = _validate_runtime_config_patch(config if isinstance(config, dict) else {})
    before = _runtime_config_snapshot()
    after = dict(before)
    after.update(validated)
    if not dry_run:
        after = _apply_runtime_config_patch(validated)
    _record_audit_event(
        action="runtime_config_profile.apply",
        resource_type="runtime_config_profile",
        resource_id=name,
        details={"dry_run": bool(dry_run), "keys": sorted(validated.keys()), "before": before, "after": after},
    )
    return {
        "profile_name": name,
        "dry_run": bool(dry_run),
        "applied": not bool(dry_run),
        "before": before,
        "after": after,
        "request_id": rid,
    }


@app.delete("/v1/admin/runtime-config/profiles/{profile_name}", dependencies=[Depends(_require_admin)])
def v1_admin_runtime_config_profile_delete(profile_name: str):
    rid = REQUEST_ID.get()
    name = _validate_runtime_profile_name(profile_name)
    deleted = RUNTIME_CONFIG_PROFILES.pop(name, None)
    if deleted is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "runtime config profile not found"})
    _persist_runtime_profiles_if_needed()
    _record_audit_event(
        action="runtime_config_profile.delete",
        resource_type="runtime_config_profile",
        resource_id=name,
    )
    return {"profile_name": name, "deleted": True, "request_id": rid}


@app.post("/v1/admin/circuit-breaker/open", dependencies=[Depends(_require_admin)])
def v1_admin_circuit_breaker_open(
    reason: str | None = None,
    duration_seconds: int | None = None,
):
    if duration_seconds is not None and int(duration_seconds) <= 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": "duration_seconds must be positive when provided"},
        )
    payload = _circuit_breaker_force_open(reason=reason, duration_seconds=duration_seconds)
    _record_audit_event(
        action="circuit_breaker.force_open",
        resource_type="circuit_breaker",
        details={"reason": payload.get("manual_reason"), "duration_seconds": duration_seconds},
    )
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.post("/v1/admin/circuit-breaker/reset", dependencies=[Depends(_require_admin)])
def v1_admin_circuit_breaker_reset():
    payload = _circuit_breaker_manual_reset()
    _record_audit_event(action="circuit_breaker.reset", resource_type="circuit_breaker")
    payload["request_id"] = REQUEST_ID.get()
    return payload


@app.get("/v1/admin/audit-logs", dependencies=[Depends(_require_admin)])
def v1_admin_audit_logs(
    limit: int = 100,
    action: str | None = None,
    resource_type: str | None = None,
    since_ts: int | None = None,
):
    rid = REQUEST_ID.get()
    rows = _query_audit_logs(limit=limit, action=action, resource_type=resource_type, since_ts=since_ts)
    return {
        "data": rows,
        "count": len(rows),
        "request_id": rid,
    }


@app.get("/v1/admin/state/export", dependencies=[Depends(_require_admin)])
def v1_admin_state_export():
    rid = REQUEST_ID.get()
    snapshot = _build_state_snapshot()
    counts = _state_counts()
    _record_audit_event(
        action="state.export",
        resource_type="state",
        details={
            "datasets": counts["datasets"],
            "eval_runs": counts["eval_runs"],
            "batch_eval_runs": counts["batch_eval_runs"],
            "runbook_runs": counts["runbook_runs"],
            "runbook_templates": counts["runbook_templates"],
            "rag_indexes": counts["rag_indexes"],
            "audit_logs": counts["audit_logs"],
            "runtime_config_profiles": len(RUNTIME_CONFIG_PROFILES),
            "slo_incidents": len(SLO_INCIDENTS),
        },
    )
    snapshot["request_id"] = rid
    return snapshot


@app.post("/v1/admin/state/import", dependencies=[Depends(_require_admin)])
def v1_admin_state_import(
    payload: Dict[str, Any] = Body(...),
    mode: str = "replace",
    dry_run: bool = False,
):
    rid = REQUEST_ID.get()
    normalized_mode = (mode or "replace").strip().lower()
    if normalized_mode not in {"replace", "merge"}:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": "mode must be replace or merge"},
        )

    extracted = _extract_snapshot_data(payload)
    datasets, eval_runs, batch_eval_runs, rag_indexes, audit_logs, runtime_config_profiles, runbook_runs, runbook_templates, maintenance_state, slo_incidents, slo_state = _normalize_loaded_state(
        extracted["datasets"],
        extracted["eval_runs"],
        extracted["batch_eval_runs"],
        extracted["rag_indexes"],
        extracted["audit_logs"],
        extracted.get("runtime_config_profiles", {}),
        extracted.get("runbook_runs", {}),
        extracted.get("runbook_templates", {}),
        extracted.get("maintenance_state", {}),
        extracted.get("slo_incidents", []),
        extracted.get("slo_state", {}),
    )
    before = _state_counts()
    incoming = {
        "datasets": len(datasets),
        "eval_runs": len(eval_runs),
        "batch_eval_runs": len(batch_eval_runs),
        "rag_indexes": len(rag_indexes),
        "audit_logs": len(audit_logs),
        "runtime_config_profiles": len(runtime_config_profiles),
        "runbook_runs": len(runbook_runs),
        "runbook_templates": len(runbook_templates),
        "maintenance_active": bool(maintenance_state.get("active", False)),
        "slo_incidents": len(slo_incidents),
    }

    if dry_run:
        return {
            "mode": normalized_mode,
            "dry_run": True,
            "before": before,
            "incoming": incoming,
            "request_id": rid,
        }

    if normalized_mode == "replace":
        DATASETS.clear()
        DATASETS.update(datasets)
        EVAL_RUNS.clear()
        EVAL_RUNS.update(eval_runs)
        BATCH_EVAL_RUNS.clear()
        BATCH_EVAL_RUNS.update(batch_eval_runs)
        RAG_INDEXES.clear()
        RAG_INDEXES.update(rag_indexes)
        AUDIT_LOGS.clear()
        AUDIT_LOGS.extend(audit_logs)
        RUNTIME_CONFIG_PROFILES.clear()
        RUNTIME_CONFIG_PROFILES.update(runtime_config_profiles)
        RUNBOOK_RUNS.clear()
        RUNBOOK_RUNS.update(runbook_runs)
        RUNBOOK_TEMPLATES.clear()
        RUNBOOK_TEMPLATES.update(runbook_templates)
        MAINTENANCE_STATE.clear()
        MAINTENANCE_STATE.update(maintenance_state)
        SLO_INCIDENTS.clear()
        SLO_INCIDENTS.extend(slo_incidents)
        SLO_STATE.clear()
        SLO_STATE.update(slo_state)
    else:
        DATASETS.update(datasets)
        EVAL_RUNS.update(eval_runs)
        BATCH_EVAL_RUNS.update(batch_eval_runs)
        RAG_INDEXES.update(rag_indexes)
        AUDIT_LOGS.extend(audit_logs)
        RUNTIME_CONFIG_PROFILES.update(runtime_config_profiles)
        RUNBOOK_RUNS.update(runbook_runs)
        RUNBOOK_TEMPLATES.update(runbook_templates)
        MAINTENANCE_STATE.update(maintenance_state)
        SLO_INCIDENTS.extend([x for x in slo_incidents if isinstance(x, dict)])
        _trim_slo_incidents()
        SLO_STATE.update(slo_state)

    _trim_audit_logs()
    _save_state_to_disk()

    after = _state_counts()
    _record_audit_event(
        action="state.import",
        resource_type="state",
        details={"mode": normalized_mode, "before": before, "incoming": incoming, "after": after},
    )
    return {
        "mode": normalized_mode,
        "dry_run": False,
        "before": before,
        "incoming": incoming,
        "after": after,
        "request_id": rid,
    }
