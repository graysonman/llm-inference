import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple, List

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.logging_utils import get_logger, log_json
from app.schemas import ChatRequest, ChatResponse, EvaluateRequest, EvaluateResponse, CriterionScore

load_dotenv()

LOGGER = get_logger()
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None
MODEL_META: Dict[str, Any] = {}


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


app = FastAPI(title="LLM Inference Server", version="0.3.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("app/static/index.html")


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
    return MODEL_META


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    rid = REQUEST_ID.get()
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
        total_tokens=total_tokens,
        output_to_input_ratio=ratio,
        refined=refined,
        original_response=original_response,
        critique=critique_text,
        refine_steps_used=steps_used,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
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

    return EvaluateResponse(
        request_id=rid,
        model=MODEL_NAME,
        scores=scores,
        latency_ms=latency_ms,
    )