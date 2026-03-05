import time

from fastapi import HTTPException

from app import main
from app.schemas import ChatRequest


def setup_function():
    main.REQUEST_ID.set("rid-chat-rt")
    main.CHAT_CACHE.clear()
    main._circuit_breaker_reset()
    with main.CHAT_CONCURRENCY_LOCK:
        main.CHAT_ACTIVE_REQUESTS = 0
    main.METRICS["chat_requests"] = 0
    main.METRICS["chat_cache_hits"] = 0
    main.METRICS["chat_cache_misses"] = 0
    main.METRICS["chat_backpressure_rejections"] = 0
    main.METRICS["chat_timeouts"] = 0
    main.MODEL_META = {
        "context_window": 2048,
        "model_type": "decoder-only",
        "attention_masking": "causal",
        "attention_heads": 8,
        "hidden_size": 256,
    }


def test_chat_backpressure_rejects_when_capacity_exhausted():
    prev_max = main.CHAT_MAX_CONCURRENT_REQUESTS
    prev_enabled = main.CIRCUIT_BREAKER_ENABLED
    main.CHAT_MAX_CONCURRENT_REQUESTS = 1
    main.CIRCUIT_BREAKER_ENABLED = False
    with main.CHAT_CONCURRENCY_LOCK:
        main.CHAT_ACTIVE_REQUESTS = 1
    try:
        try:
            main.chat(ChatRequest(prompt="blocked", max_new_tokens=8))
            assert False, "expected backpressure rejection"
        except HTTPException as exc:
            assert exc.status_code == 503
            detail = exc.detail
            assert detail["code"] == "service_unavailable"
            assert detail["details"]["reason"] == "backpressure"
        assert main.METRICS["chat_backpressure_rejections"] >= 1
    finally:
        with main.CHAT_CONCURRENCY_LOCK:
            main.CHAT_ACTIVE_REQUESTS = 0
        main.CHAT_MAX_CONCURRENT_REQUESTS = prev_max
        main.CIRCUIT_BREAKER_ENABLED = prev_enabled


def test_chat_timeout_budget_enforced():
    prev_timeout = main.CHAT_REQUEST_TIMEOUT_MS
    prev_enabled = main.CIRCUIT_BREAKER_ENABLED
    prev_generate = main._generate
    main.CHAT_REQUEST_TIMEOUT_MS = 60
    main.CIRCUIT_BREAKER_ENABLED = False

    def _slow_generate(*args, **kwargs):
        time.sleep(0.09)
        return "ok", 1, 1

    main._generate = _slow_generate
    try:
        try:
            main.chat(ChatRequest(prompt="timeout me", max_new_tokens=8))
            assert False, "expected timeout budget rejection"
        except HTTPException as exc:
            assert exc.status_code == 503
            detail = exc.detail
            assert detail["code"] == "service_unavailable"
            assert detail["details"]["reason"] == "timeout"
        assert main.METRICS["chat_timeouts"] >= 1
    finally:
        main.CHAT_REQUEST_TIMEOUT_MS = prev_timeout
        main.CIRCUIT_BREAKER_ENABLED = prev_enabled
        main._generate = prev_generate

