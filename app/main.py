import csv
import hashlib
import io
import json
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

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
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

tokenizer = None
model = None
MODEL_META: Dict[str, Any] = {}
CHAT_CACHE: Dict[str, Dict[str, Any]] = {}
METRICS: Dict[str, int] = {
    "chat_requests": 0,
    "chat_cache_hits": 0,
    "chat_cache_misses": 0,
}
DATASETS: Dict[str, Dict[str, Any]] = {}
EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
BATCH_EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
RAG_INDEXES: Dict[str, Dict[str, Any]] = {}
RATE_LIMIT_BUCKETS: Dict[str, Dict[str, int]] = {}
BATCH_EVAL_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()
CACHE_BACKEND = "in_memory"
REDIS_CLIENT = None
ROLE_SCOPE_DEFAULTS: Dict[str, List[str]] = {
    "admin": ["*"],
    "analyst": [
        "chat:invoke",
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
API_KEY_REGISTRY: Dict[str, Dict[str, Any]] = {}
RETENTION_THREAD: threading.Thread | None = None
RETENTION_STOP_EVENT = threading.Event()
RETENTION_STATS: Dict[str, Any] = {
    "last_run_ts": None,
    "last_purged_total": 0,
    "last_error": None,
}
RETENTION_HISTORY: List[Dict[str, Any]] = []


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

    return auth


def _state_backend_name() -> str:
    if not STATE_PERSISTENCE_ENABLED:
        return "in_memory"
    if STATE_BACKEND == "sqlite":
        return f"sqlite:{STATE_SQLITE_PATH}"
    return f"file_json:{STATE_FILE_PATH}"


def _normalize_loaded_state(
    datasets: Any,
    eval_runs: Any,
    batch_eval_runs: Any,
    rag_indexes: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not isinstance(datasets, dict):
        datasets = {}
    if not isinstance(eval_runs, dict):
        eval_runs = {}
    if not isinstance(batch_eval_runs, dict):
        batch_eval_runs = {}
    if not isinstance(rag_indexes, dict):
        rag_indexes = {}

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

    return datasets, eval_runs, batch_eval_runs, rag_indexes


def _save_state_json() -> None:
    state = {
        "datasets": DATASETS,
        "eval_runs": EVAL_RUNS,
        "batch_eval_runs": BATCH_EVAL_RUNS,
        "rag_indexes": RAG_INDEXES,
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

    datasets, eval_runs, batch_eval_runs, rag_indexes = _normalize_loaded_state(
        loaded.get("datasets", {}),
        loaded.get("eval_runs", {}),
        loaded.get("batch_eval_runs", {}),
        loaded.get("rag_indexes", {}),
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

    datasets, eval_runs, batch_eval_runs, rag_indexes = _normalize_loaded_state(
        datasets_loaded,
        eval_runs_loaded,
        batch_eval_runs_loaded,
        rag_indexes_loaded,
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
        finally:
            conn.close()
        return {
            "datasets": datasets_count,
            "eval_runs": eval_runs_count,
            "batch_eval_runs": batch_runs_count,
            "rag_indexes": rag_indexes_count,
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


def _mark_batch_eval_failed(run_id: str, message: str) -> None:
    run_snapshot: Optional[Dict[str, Any]] = None
    with BATCH_EVAL_LOCK:
        run = BATCH_EVAL_RUNS.get(run_id)
        if run is None:
            return
        total = int(run.get("progress", {}).get("total", 0))
        run["status"] = "failed"
        run["progress"] = _build_batch_progress(total=total, completed=0, failed=total)
        run["summary"] = {
            "mean_scores": {},
            "total_items": total,
            "failed_items": total,
            "error": message,
        }
        run["record_scores"] = []
        run["failures"] = []
        run["updated_at"] = _now_ts()
        run["completed_at"] = _now_ts()
        run_snapshot = dict(run)
    if run_snapshot is not None:
        _persist_batch_eval_run_record(run_snapshot)


def _run_batch_eval_worker(run_id: str) -> None:
    run_snapshot: Optional[Dict[str, Any]] = None
    with BATCH_EVAL_LOCK:
        run = BATCH_EVAL_RUNS.get(run_id)
        if run is None or run.get("status") != "queued":
            return
        run["status"] = "running"
        run["started_at"] = _now_ts()
        run["updated_at"] = _now_ts()
        dataset_id = run["dataset_id"]
        criteria = list(run.get("criteria", []))
        model_name = run.get("model", MODEL_NAME)
        run_snapshot = dict(run)
    if run_snapshot is not None:
        _persist_batch_eval_run_record(run_snapshot)

    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        _mark_batch_eval_failed(run_id, "dataset_id not found")
        return

    records = list(dataset.get("records", []))
    total = len(records)
    record_scores: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    failed_records = 0

    for idx, record in enumerate(records):
        row_scores = {criterion: _score_record(record, criterion) for criterion in criteria}
        record_scores.append({"record_index": idx, "record": record, "scores": row_scores})

        row_failed = False
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
            run["progress"] = _build_batch_progress(total=total, completed=idx + 1, failed=failed_records)
            run["updated_at"] = _now_ts()
            run_snapshot = dict(run)
        if idx == total - 1 or ((idx + 1) % 10 == 0):
            _persist_batch_eval_run_record(run_snapshot)

    summary = _compute_batch_summary(records, criteria)
    now = _now_ts()
    with BATCH_EVAL_LOCK:
        run = BATCH_EVAL_RUNS.get(run_id)
        if run is None:
            return
        run["status"] = "completed"
        run["summary"] = summary
        run["model"] = model_name
        run["progress"] = _build_batch_progress(total=total, completed=total, failed=summary.get("failed_items", 0))
        run["updated_at"] = now
        run["completed_at"] = now
        run_snapshot = dict(run)
    if run_snapshot is not None:
        _persist_batch_eval_run_record(run_snapshot)

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


def _render_metrics_text() -> str:
    cache = _cache_stats()
    counts = _state_counts()
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
        "# HELP rate_limit_rejections_total Number of rate-limited requests",
        "# TYPE rate_limit_rejections_total counter",
        f"rate_limit_rejections_total {sum(max(v['count'] - RATE_LIMIT_PER_MINUTE, 0) for v in RATE_LIMIT_BUCKETS.values())}",
    ]
    return "\n".join(lines) + "\n"


def _startup_impl() -> None:
    global tokenizer, model, MODEL_META, RETENTION_THREAD

    _init_cache_backend()
    _init_auth_registry()
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


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    REQUEST_ID.set(rid)

    start = time.perf_counter()

    rate_meta = None
    if _is_protected_path(str(request.url.path)):
        bucket_key = _resolve_api_key(request.headers.get("x-api-key"), request.headers.get("authorization")) or "anonymous"
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


@app.get("/readyz")
def readyz():
    ready = tokenizer is not None and model is not None
    return {
        "status": "ready" if ready else "not_ready",
        "model_loaded": ready,
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

    context_window = MODEL_META["context_window"]
    model_type = MODEL_META["model_type"]
    masking = MODEL_META["attention_masking"]
    heads = MODEL_META["attention_heads"]
    hidden = MODEL_META["hidden_size"]

    # Run generation
    start = time.perf_counter()
    response_text, prompt_tokens, completion_tokens = _generate(
        prompt=user_prompt,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
    )

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

            critique_prompt = _build_critique_prompt(user_prompt, current_answer)
            critique_text, _, _ = _generate(
                prompt=critique_prompt,
                max_new_tokens=min(256, payload.max_new_tokens),
                temperature=payload.critique_temperature,
                top_p=payload.top_p,
            )

            refine_prompt = _build_refine_prompt(user_prompt, current_answer, critique_text)
            improved_answer, _, _ = _generate(
                prompt=refine_prompt,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )

            current_answer = improved_answer

        # Final answer is the refined one
        response_text = current_answer

        # Update token counts for final prompt/answer visibility
        # Keep the original prompt_tokens as the user prompt tokens, but recompute completion tokens from final answer length
        # For this stage, we track totals at the API boundary rather than exact per-pass accounting.
        completion_tokens = _count_tokens(response_text)

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

    sample_text = "No indexed chunks available yet."
    if dataset["records"]:
        first = dataset["records"][0]
        if isinstance(first.get("input"), dict):
            sample_text = str(first["input"].get("prompt") or first)
        else:
            sample_text = str(first.get("text") or first.get("input") or first)

    return RagContractResponse(
        answer="RAG retrieval skeleton response. Full retrieval/generation pipeline is pending.",
        citations=[
            {
                "doc_id": dataset["dataset_id"],
                "chunk_id": f"{payload.dataset_id}-chunk-1",
                "score": 0.42,
                "text": sample_text,
            }
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

    sample_text = "No indexed chunks available yet."
    if dataset["records"]:
        first = dataset["records"][0]
        sample_text = first.get("text") or first.get("input") or str(first)

    return RagQueryResponse(
        response="RAG retrieval skeleton response. Full retrieval/generation pipeline is pending.",
        retrieved_chunks=[
            {
                "chunk_id": f"{payload.dataset_id}-chunk-1",
                "dataset_id": payload.dataset_id,
                "score": 0.42,
                "text": sample_text,
            }
        ],
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
    RAG_INDEXES[dataset_id] = {
        "index_id": dataset_id,
        "dataset_id": dataset_id,
        "status": "ready",
        "chunk_count": len(payload.records),
        "updated_at": now,
        "deleted_at": None,
    }
    _persist_dataset_record(DATASETS[dataset_id])
    _persist_rag_index_record(RAG_INDEXES[dataset_id])
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
    RAG_INDEXES[dataset_id] = {
        "index_id": dataset_id,
        "dataset_id": dataset_id,
        "status": "ready",
        "chunk_count": len(accepted),
        "updated_at": now,
        "deleted_at": None,
    }
    _persist_dataset_record(DATASETS[dataset_id])
    _persist_rag_index_record(RAG_INDEXES[dataset_id])

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
    return {"dataset_id": dataset_id, "deleted": True, "request_id": rid}


@app.post("/v1/datasets/{dataset_id}/restore", dependencies=[Depends(_require_datasets_write)])
def v1_restore_dataset(dataset_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_dataset_record(dataset_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset not found"})
    return {"dataset_id": dataset_id, "restored": True, "request_id": rid}


@app.delete("/v1/datasets/{dataset_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_dataset(dataset_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_dataset_record(dataset_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset not found"})
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

    with BATCH_EVAL_LOCK:
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
        }
        run_snapshot = dict(BATCH_EVAL_RUNS[run_id])
    _persist_batch_eval_run_record(run_snapshot)

    worker = threading.Thread(target=_run_batch_eval_worker, args=(run_id,), daemon=True, name=f"batch-eval-{run_id}")
    worker.start()
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
    return {"run_id": run_id, "deleted": True, "request_id": rid}


@app.post("/v1/evals/{run_id}/restore", dependencies=[Depends(_require_evals_write)])
def v1_restore_eval_run(run_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_eval_run_record(run_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "eval run not found"})
    return {"run_id": run_id, "restored": True, "request_id": rid}


@app.delete("/v1/evals/{run_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_eval_run(run_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_eval_run_record(run_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "eval run not found"})
    return {"run_id": run_id, "purged": True, "request_id": rid}


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
    return {"run_id": run_id, "deleted": True, "request_id": rid}


@app.post("/v1/batch-evals/{run_id}/restore", dependencies=[Depends(_require_batch_write)])
def v1_restore_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_batch_eval_run_record(run_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
    return {"run_id": run_id, "restored": True, "request_id": rid}


@app.delete("/v1/batch-evals/{run_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_batch_eval_run_record(run_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})
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

@app.delete("/v1/rag/indexes/{index_id}", dependencies=[Depends(_require_rag_write)])
def v1_delete_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    deleted = _delete_rag_index_record(index_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})
    return {"index_id": index_id, "deleted": True, "request_id": rid}


@app.post("/v1/rag/indexes/{index_id}/restore", dependencies=[Depends(_require_rag_write)])
def v1_restore_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    restored = _restore_rag_index_record(index_id)
    if not restored:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})
    return {"index_id": index_id, "restored": True, "request_id": rid}


@app.delete("/v1/rag/indexes/{index_id}/purge", dependencies=[Depends(_require_admin)])
def v1_purge_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    purged = _purge_rag_index_record(index_id)
    if not purged:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "rag index not found"})
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


@app.get("/v1/metrics", dependencies=[Depends(_require_metrics_read)])
def v1_metrics(format: str = "json"):
    if format in {"text", "prometheus", "openmetrics"}:
        return PlainTextResponse(_render_metrics_text(), media_type="text/plain; version=0.0.4")
    counts = _state_counts()

    return {
        "request_id": REQUEST_ID.get(),
        "model": MODEL_NAME,
        "chat": {
            "requests": METRICS["chat_requests"],
        },
        "datasets": {"count": counts["datasets"]},
        "eval_runs": {"count": counts["eval_runs"]},
        "batch_eval_runs": {"count": counts["batch_eval_runs"]},
        "cache": _cache_stats(),
        "rate_limit": {
            "requests_per_minute_per_key": RATE_LIMIT_PER_MINUTE,
            "active_buckets": len(RATE_LIMIT_BUCKETS),
        },
        "persistence": {
            "backend": _state_backend_name(),
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


@app.get("/metrics/dashboard", dependencies=[Depends(_require_metrics_read)])
def metrics_dashboard(window: str = "15m"):
    cache = _cache_stats()
    counts = _state_counts()
    queue_depth = _batch_queue_depth()
    total_chat = METRICS["chat_requests"]
    rejected = sum(max(v["count"] - RATE_LIMIT_PER_MINUTE, 0) for v in RATE_LIMIT_BUCKETS.values())

    degraded = tokenizer is None or model is None

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
        },
        "cache": cache,
        "latency": {
            "note": "Use x-latency-ms response header and external scrape for p50/p95/p99 trends.",
        },
        "errors": {
            "rate_limited": rejected,
            "retention_last_error": RETENTION_STATS.get("last_error"),
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
    result["request_id"] = REQUEST_ID.get()
    return result
