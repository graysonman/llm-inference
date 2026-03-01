import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple, List

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("API_KEY", "dev-local-key")
CACHE_TTL_SECONDS = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "120"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

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
    prompt_tokens = _count_tokens(prompt)

    context_window = MODEL_META["context_window"]
    _enforce_context_guardrail(prompt_tokens, max_new_tokens, context_window)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    do_sample = temperature > 0.0

    try:
        with torch.inference_mode():
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


def _get_cached_chat_response(cache_key: str) -> Optional[ChatResponse]:
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
    CHAT_CACHE[cache_key] = {
        "created_at": time.time(),
        "response": response.model_dump(),
    }


def _cache_stats() -> Dict[str, Any]:
    now = time.time()
    active_keys = []
    for key, value in CHAT_CACHE.items():
        if now - value["created_at"] <= CACHE_TTL_SECONDS:
            active_keys.append(key)
    for key in set(CHAT_CACHE.keys()) - set(active_keys):
        CHAT_CACHE.pop(key, None)

    hits = METRICS["chat_cache_hits"]
    misses = METRICS["chat_cache_misses"]
    total = hits + misses
    hit_rate = round(hits / total, 4) if total else 0.0

    return {
        "ttl_seconds": CACHE_TTL_SECONDS,
        "entries": len(active_keys),
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
    }


def _now_ts() -> int:
    return int(time.time())


def _build_batch_progress(total: int) -> Dict[str, int]:
    return {"total": total, "completed": total, "failed": 0}


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
        "# HELP datasets_total Number of datasets stored in memory",
        "# TYPE datasets_total gauge",
        f"datasets_total {len(DATASETS)}",
        "# HELP eval_runs_total Number of eval runs stored in memory",
        "# TYPE eval_runs_total gauge",
        f"eval_runs_total {len(EVAL_RUNS)}",
        "# HELP batch_eval_runs_total Number of batch eval runs stored in memory",
        "# TYPE batch_eval_runs_total gauge",
        f"batch_eval_runs_total {len(BATCH_EVAL_RUNS)}",
        "# HELP rate_limit_rejections_total Number of rate-limited requests",
        "# TYPE rate_limit_rejections_total counter",
        f"rate_limit_rejections_total {sum(max(v['count'] - RATE_LIMIT_PER_MINUTE, 0) for v in RATE_LIMIT_BUCKETS.values())}",
    ]
    return "\n".join(lines) + "\n"


app = FastAPI(title="LLM Inference Server", version="0.3.0")

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


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API key"})


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
        bucket_key = request.headers.get("x-api-key") or "anonymous"
        rate_meta = _apply_rate_limit(bucket_key)
        if rate_meta["limited"]:
            details = {
                "limit": rate_meta["limit"],
                "remaining": rate_meta["remaining"],
                "reset_at": rate_meta["reset_at"],
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
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "request_id": rid})


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error_response(
        400,
        "invalid_request",
        "Request validation failed",
        {"errors": exc.errors()},
    )


@app.on_event("startup")
def startup():
    global tokenizer, model, MODEL_META

    log_json(LOGGER, {"event": "startup", "model_name": MODEL_NAME, "device": DEVICE})

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
        "cache": "in_memory",
        "persistence": "in_memory",
        "request_id": REQUEST_ID.get(),
    }


@app.get("/model")
def model_info():
    return MODEL_META


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
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


@app.post("/evaluate", response_model=EvaluateResponse, dependencies=[Depends(_require_api_key)])
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
    }

    return EvaluateResponse(
        request_id=rid,
        model=MODEL_NAME,
        scores=scores,
        latency_ms=latency_ms,
        run_id=run_id,
    )


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
def v1_chat(payload: ChatRequest):
    return chat(payload)


@app.post("/v1/evaluate", response_model=EvaluateResponse, dependencies=[Depends(_require_api_key)])
def v1_evaluate(payload: EvaluateRequest):
    return evaluate(payload)


@app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=[Depends(_require_api_key)])
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


@app.post("/v1/rag/query", response_model=RagQueryResponse, dependencies=[Depends(_require_api_key)])
def v1_rag_query(payload: RagQueryRequest):
    rid = REQUEST_ID.get()
    dataset = DATASETS.get(payload.dataset_id)
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


@app.post("/v1/datasets", response_model=DatasetCreateResponse, status_code=201, dependencies=[Depends(_require_api_key)])
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
    }
    RAG_INDEXES[dataset_id] = {
        "index_id": dataset_id,
        "dataset_id": dataset_id,
        "status": "ready",
        "chunk_count": len(payload.records),
        "updated_at": now,
    }
    return DatasetCreateResponse(
        dataset_id=dataset_id,
        name=payload.name,
        type=payload.type,
        status="ready",
        records_count=len(payload.records),
        created_at=now,
        request_id=rid,
    )


@app.get("/v1/datasets", response_model=DatasetListResponse, dependencies=[Depends(_require_api_key)])
def v1_list_datasets(
    type: str | None = None,
    status: str | None = None,
    limit: int = 20,
):
    rid = REQUEST_ID.get()
    clamped_limit = max(1, min(limit, 100))
    items: List[DatasetListItem] = []
    for ds in DATASETS.values():
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


@app.get("/v1/datasets/{dataset_id}", response_model=DatasetGetResponse, dependencies=[Depends(_require_api_key)])
def v1_get_dataset(dataset_id: str):
    rid = REQUEST_ID.get()
    ds = DATASETS.get(dataset_id)
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


@app.post("/v1/batch-evals", response_model=BatchEvalCreateResponse, status_code=202, dependencies=[Depends(_require_api_key)])
def v1_create_batch_eval(payload: BatchEvalCreateRequest):
    rid = REQUEST_ID.get()
    dataset = DATASETS.get(payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "dataset_id not found"})

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    now = _now_ts()
    progress = _build_batch_progress(len(dataset["records"]))
    criteria = [c for c in payload.criteria]
    summary = _compute_batch_summary(dataset["records"], criteria)
    record_scores = _build_batch_record_scores(dataset["records"], criteria)
    failures = []
    for row in record_scores:
        for criterion_name, score in row["scores"].items():
            if score <= 3:
                failures.append({
                    "record_index": row["record_index"],
                    "criterion": criterion_name,
                    "score": score,
                    "reason": f"low score for {criterion_name}",
                    "record": row["record"],
                })
    BATCH_EVAL_RUNS[run_id] = {
        "run_id": run_id,
        "dataset_id": payload.dataset_id,
        "status": "completed",
        "progress": progress,
        "criteria": payload.criteria,
        "summary": summary,
        "record_scores": record_scores,
        "failures": failures,
        "started_at": now,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    return BatchEvalCreateResponse(run_id=run_id, request_id=rid)


@app.get("/v1/evals/{run_id}", response_model=EvalRunSummary, dependencies=[Depends(_require_api_key)])
def v1_get_eval_run(run_id: str):
    run = EVAL_RUNS.get(run_id)
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


@app.get("/v1/batch-evals/{run_id}", response_model=BatchEvalStatusResponse, dependencies=[Depends(_require_api_key)])
def v1_get_batch_eval(run_id: str):
    rid = REQUEST_ID.get()
    run = BATCH_EVAL_RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    return BatchEvalStatusResponse(
        run_id=run["run_id"],
        dataset_id=run["dataset_id"],
        status=run["status"],
        progress=run["progress"],
        criteria=run["criteria"],
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        request_id=rid,
    )


@app.get("/v1/rag/indexes/{index_id}", response_model=RagIndexStatusResponse, dependencies=[Depends(_require_api_key)])
def v1_get_rag_index(index_id: str):
    rid = REQUEST_ID.get()
    idx = RAG_INDEXES.get(index_id)
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


@app.get("/v1/batch-evals/{run_id}/result", response_model=BatchEvalResultResponse, dependencies=[Depends(_require_api_key)])
def v1_get_batch_eval_result(run_id: str):
    rid = REQUEST_ID.get()
    run = BATCH_EVAL_RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    return BatchEvalResultResponse(
        batch_eval_id=run["run_id"],
        status=run["status"],
        summary=run.get("summary", {"mean_scores": {}, "total_items": 0, "failed_items": 0}),
        completed_at=run.get("completed_at"),
        request_id=rid,
    )


@app.get("/v1/batch-evals/{run_id}/failures", response_model=BatchEvalFailuresResponse, dependencies=[Depends(_require_api_key)])
def v1_get_batch_eval_failures(run_id: str, limit: int = 20):
    rid = REQUEST_ID.get()
    run = BATCH_EVAL_RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "batch eval run not found"})

    clamped_limit = max(1, min(limit, 100))
    data = [BatchEvalFailureItem.model_validate(item) for item in run.get("failures", [])[:clamped_limit]]
    return BatchEvalFailuresResponse(
        batch_eval_id=run["run_id"],
        data=data,
        count=len(run.get("failures", [])),
        request_id=rid,
    )


@app.get("/v1/batch-evals/{run_id}/distribution", response_model=BatchEvalDistributionResponse, dependencies=[Depends(_require_api_key)])
def v1_get_batch_eval_distribution(run_id: str, criterion: str = "overall"):
    rid = REQUEST_ID.get()
    run = BATCH_EVAL_RUNS.get(run_id)
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
        batch_eval_id=run["run_id"],
        criterion=criterion,
        buckets=dist["buckets"],
        summary=dist["summary"],
        request_id=rid,
    )


@app.get("/v1/metrics", dependencies=[Depends(_require_api_key)])
def v1_metrics(format: str = "json"):
    if format in {"text", "prometheus", "openmetrics"}:
        return PlainTextResponse(_render_metrics_text(), media_type="text/plain; version=0.0.4")

    return {
        "request_id": REQUEST_ID.get(),
        "model": MODEL_NAME,
        "chat": {
            "requests": METRICS["chat_requests"],
        },
        "datasets": {"count": len(DATASETS)},
        "eval_runs": {"count": len(EVAL_RUNS)},
        "batch_eval_runs": {"count": len(BATCH_EVAL_RUNS)},
        "cache": _cache_stats(),
        "rate_limit": {
            "requests_per_minute_per_key": RATE_LIMIT_PER_MINUTE,
            "active_buckets": len(RATE_LIMIT_BUCKETS),
        },
    }
