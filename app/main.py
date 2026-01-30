import os
import time
import uuid
from contextvars import ContextVar
from typing import Tuple

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

# ---- Config ----
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Globals initialized on startup ----
tokenizer = None
model = None


def _count_tokens(text: str) -> int:
    # HF tokenizers return input_ids tensor-like; use length
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return int(ids.numel())


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Tuple[str, int, int]:
    """
    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    prompt_tokens = _count_tokens(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Determinism-ish: temperature low by default, do_sample True only if temp > 0
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

    # output_ids includes prompt + completion for causal LM
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Heuristic: completion is the suffix beyond the original prompt length in tokens
    # More reliable than string slicing when tokenization differs.
    completion_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])

    response_text = full_text.strip()

    return response_text, prompt_tokens, completion_tokens


app = FastAPI(title="Baseline LLM Inference Server", version="0.1.0")


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
    global tokenizer, model

    log_json(
        LOGGER,
        {
            "event": "startup",
            "model_name": MODEL_NAME,
            "device": DEVICE,
        },
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        log_json(LOGGER, {"event": "startup_failed", "error": repr(e)})
        raise


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    rid = REQUEST_ID.get()
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    start = time.perf_counter()
    try:
        response_text, prompt_tokens, completion_tokens = _generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_failed: {repr(e)}") from e

    latency_ms = int((time.perf_counter() - start) * 1000)

    log_json(
        LOGGER,
        {
            "event": "chat",
            "request_id": rid,
            "model": MODEL_NAME,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
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
    )
