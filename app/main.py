import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.logging_utils import get_logger, log_json
from app.schemas import ChatRequest, ChatResponse

load_dotenv()

LOGGER = get_logger()
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

# Cached model metadata for reuse
MODEL_META: Dict[str, Any] = {}


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _get_model_type() -> str:
    """
    Best-effort identification:
    - decoder-only: causal LM (GPT-like)
    - encoder-decoder: seq2seq (T5/BART-like)
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "unknown"
    if getattr(cfg, "is_encoder_decoder", False):
        return "encoder-decoder"
    # Our server loads AutoModelForCausalLM, so this should be decoder-only
    return "decoder-only"


def _get_attention_masking() -> str:
    """
    For decoder-only generation, attention is typically causal (future-masked).
    We expose this explicitly because it maps to your notes: 'softmax to 0' for forbidden edges.
    """
    mt = _get_model_type()
    if mt == "decoder-only":
        return "causal"
    if mt == "encoder-decoder":
        return "unknown"
    return "unknown"


def _get_context_window() -> int:
    """
    Tries multiple config fields because different models use different names.
    Falls back to tokenizer.model_max_length when it's sane.
    """
    cfg = getattr(model, "config", None)

    candidates = []
    if cfg is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            v = getattr(cfg, attr, None)
            iv = _safe_int(v)
            if iv:
                candidates.append(iv)

    # tokenizer.model_max_length is sometimes an absurd sentinel (e.g. 1000000000000)
    tok_max = _safe_int(getattr(tokenizer, "model_max_length", None))
    if tok_max and tok_max < 100_000:
        candidates.append(tok_max)

    # Choose the smallest non-trivial candidate (safer)
    candidates = [c for c in candidates if c >= 128]
    if not candidates:
        # reasonable fallback
        return 2048
    return min(candidates)


def _count_tokens(text: str) -> int:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return int(ids.numel())


def _get_heads_and_hidden() -> Tuple[Optional[int], Optional[int]]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None, None

    # Common names across model families
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
    # Illustrative metric to map attention to "message passing on a fully connected graph"
    # Not exact FLOPs; just n^2 to communicate scaling clearly.
    return int(seq_len) * int(seq_len)


def _enforce_context_guardrail(prompt_tokens: int, max_new_tokens: int, context_window: int) -> None:
    """
    For decoder-only models: the total tokens (prompt + generated) must fit within context window.
    We enforce it explicitly so the system boundary reflects architectural constraints.
    """
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
    """
    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    prompt_tokens = _count_tokens(prompt)

    context_window = MODEL_META["context_window"]
    _enforce_context_guardrail(prompt_tokens, max_new_tokens, context_window)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    do_sample = temperature > 0.0

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    completion_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])

    # Keep it simple and robust: return full decoded text if stripping fails.
    # Some chat models have special tokens that make naive prefix stripping unreliable.
    if full_text.startswith(prompt):
        response_text = full_text[len(prompt):].lstrip()
    else:
        response_text = full_text.strip()

    return response_text, prompt_tokens, completion_tokens


app = FastAPI(title="Baseline LLM Inference Server", version="0.2.0")


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    REQUEST_ID.set(rid)

    start = time.perf_counter()
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
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "request_id": rid})


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


@app.get("/model")
def model_info():
    """
    Small introspection endpoint so you can explain:
    - decoder-only vs encoder-decoder
    - causal masking
    - context window
    - heads / hidden size
    """
    return MODEL_META


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    rid = REQUEST_ID.get()
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    # Mechanics computed up-front (maps to your Video 2 notes)
    context_window = MODEL_META["context_window"]
    model_type = MODEL_META["model_type"]
    masking = MODEL_META["attention_masking"]
    heads = MODEL_META["attention_heads"]
    hidden = MODEL_META["hidden_size"]

    prompt_tokens = _count_tokens(prompt)

    # Guardrail BEFORE generation
    _enforce_context_guardrail(prompt_tokens, payload.max_new_tokens, context_window)

    start = time.perf_counter()
    try:
        response_text, prompt_tokens2, completion_tokens = _generate(
            prompt=prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        # prompt_tokens2 should equal prompt_tokens, but keep the generated value for consistency
        prompt_tokens = prompt_tokens2
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_failed: {repr(e)}") from e

    latency_ms = int((time.perf_counter() - start) * 1000)

    total_tokens_effective = prompt_tokens + completion_tokens
    context_used_pct = round((total_tokens_effective / context_window) * 100.0, 4)
    est_ops = _estimated_attention_ops(total_tokens_effective)

    log_json(
        LOGGER,
        {
            "event": "chat",
            "request_id": rid,
            "model": MODEL_NAME,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens_effective,
            "context_window": context_window,
            "context_used_pct": context_used_pct,
            "model_type": model_type,
            "attention_masking": masking,
            "attention_heads": heads,
            "hidden_size": hidden,
            "estimated_attention_ops": est_ops,
            "max_new_tokens": payload.max_new_tokens,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
        },
    )

    return ChatResponse(
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
    )