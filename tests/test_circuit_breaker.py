from fastapi import HTTPException

from app import main
from app.schemas import ChatRequest


def setup_function():
    main.REQUEST_ID.set("rid-cb")
    main.CHAT_CACHE.clear()
    main.METRICS["chat_requests"] = 0
    main.METRICS["chat_cache_hits"] = 0
    main.METRICS["chat_cache_misses"] = 0
    main._circuit_breaker_reset()


def _set_min_model_meta():
    main.MODEL_META = {
        "context_window": 2048,
        "model_type": "decoder-only",
        "attention_masking": "causal",
        "attention_heads": 8,
        "hidden_size": 256,
    }


def test_circuit_breaker_opens_and_blocks_chat():
    prev_enabled = main.CIRCUIT_BREAKER_ENABLED
    prev_threshold = main.CIRCUIT_BREAKER_FAILURE_THRESHOLD
    prev_cooldown = main.CIRCUIT_BREAKER_COOLDOWN_SECONDS
    prev_generate = main._generate
    prev_tokenizer = main.tokenizer
    prev_model = main.model
    main.CIRCUIT_BREAKER_ENABLED = True
    main.CIRCUIT_BREAKER_FAILURE_THRESHOLD = 2
    main.CIRCUIT_BREAKER_COOLDOWN_SECONDS = 30
    _set_min_model_meta()
    main.tokenizer = object()
    main.model = object()

    def _fail_generate(*args, **kwargs):
        raise HTTPException(status_code=400, detail={"error": "generation_failed", "detail": "simulated"})

    main._generate = _fail_generate
    try:
        for _ in range(2):
            try:
                main.chat(ChatRequest(prompt="cb fail", max_new_tokens=8))
                assert False, "expected generation failure"
            except HTTPException as exc:
                assert exc.status_code == 400

        try:
            main.chat(ChatRequest(prompt="cb blocked", max_new_tokens=8))
            assert False, "expected circuit-open rejection"
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail.get("code") == "service_unavailable"

        readiness = main.readyz()
        assert readiness["status"] == "not_ready"
        assert readiness["circuit_breaker"]["state"] == "open"
    finally:
        main.CIRCUIT_BREAKER_ENABLED = prev_enabled
        main.CIRCUIT_BREAKER_FAILURE_THRESHOLD = prev_threshold
        main.CIRCUIT_BREAKER_COOLDOWN_SECONDS = prev_cooldown
        main._generate = prev_generate
        main.tokenizer = prev_tokenizer
        main.model = prev_model
        main._circuit_breaker_reset()


def test_circuit_breaker_half_open_recovers_on_success():
    prev_enabled = main.CIRCUIT_BREAKER_ENABLED
    prev_threshold = main.CIRCUIT_BREAKER_FAILURE_THRESHOLD
    prev_cooldown = main.CIRCUIT_BREAKER_COOLDOWN_SECONDS
    prev_generate = main._generate
    main.CIRCUIT_BREAKER_ENABLED = True
    main.CIRCUIT_BREAKER_FAILURE_THRESHOLD = 1
    main.CIRCUIT_BREAKER_COOLDOWN_SECONDS = 1
    _set_min_model_meta()

    def _fail_generate(*args, **kwargs):
        raise HTTPException(status_code=400, detail={"error": "generation_failed", "detail": "simulated"})

    def _ok_generate(*args, **kwargs):
        return "ok", 1, 1

    main._generate = _fail_generate
    try:
        try:
            main.chat(ChatRequest(prompt="trip", max_new_tokens=8))
            assert False, "expected generation failure"
        except HTTPException:
            pass

        snap = main._circuit_breaker_snapshot()
        assert snap["state"] == "open"

        with main.CIRCUIT_BREAKER_LOCK:
            main.CIRCUIT_BREAKER["opened_at"] = main._now_ts() - 2

        main._generate = _ok_generate
        out = main.chat(ChatRequest(prompt="recover", max_new_tokens=8))
        assert out.response == "ok"
        assert main._circuit_breaker_snapshot()["state"] == "closed"
    finally:
        main.CIRCUIT_BREAKER_ENABLED = prev_enabled
        main.CIRCUIT_BREAKER_FAILURE_THRESHOLD = prev_threshold
        main.CIRCUIT_BREAKER_COOLDOWN_SECONDS = prev_cooldown
        main._generate = prev_generate
        main._circuit_breaker_reset()


def test_metrics_include_circuit_breaker_fields():
    payload = main.v1_metrics(format="json")
    assert "circuit_breaker" in payload
    text = main.v1_metrics(format="prometheus").body.decode("utf-8")
    assert "circuit_breaker_state " in text
    assert "circuit_breaker_consecutive_failures " in text

