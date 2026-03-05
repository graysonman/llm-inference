from fastapi import HTTPException

from app import main
from app.schemas import ChatRequest


def setup_function():
    main.REQUEST_ID.set("rid-cb-admin")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.CHAT_CACHE.clear()
    main.AUDIT_LOGS.clear()
    main._circuit_breaker_reset()
    main.MODEL_META = {
        "context_window": 2048,
        "model_type": "decoder-only",
        "attention_masking": "causal",
        "attention_heads": 8,
        "hidden_size": 256,
    }


def test_admin_force_open_blocks_chat_and_reset_recovers():
    prev_generate = main._generate
    prev_enabled = main.CIRCUIT_BREAKER_ENABLED
    main.CIRCUIT_BREAKER_ENABLED = True

    def _ok_generate(*args, **kwargs):
        return "ok", 1, 1

    main._generate = _ok_generate
    try:
        opened = main.v1_admin_circuit_breaker_open(reason="maintenance", duration_seconds=60)
        assert opened["state"] == "open"
        assert opened["manual_forced_open"] is True
        assert opened["manual_reason"] == "maintenance"

        try:
            main.chat(ChatRequest(prompt="should block", max_new_tokens=8))
            assert False, "expected service_unavailable while manually open"
        except HTTPException as exc:
            assert exc.status_code == 503

        reset = main.v1_admin_circuit_breaker_reset()
        assert reset["state"] == "closed"
        assert reset["manual_forced_open"] is False

        out = main.chat(ChatRequest(prompt="allowed after reset", max_new_tokens=8))
        assert out.response == "ok"

        actions = [row.get("action") for row in main.AUDIT_LOGS]
        assert "circuit_breaker.force_open" in actions
        assert "circuit_breaker.reset" in actions
    finally:
        main._generate = prev_generate
        main.CIRCUIT_BREAKER_ENABLED = prev_enabled
        main._circuit_breaker_reset()


def test_admin_force_open_duration_expiry_releases_block():
    prev_generate = main._generate
    prev_enabled = main.CIRCUIT_BREAKER_ENABLED
    main.CIRCUIT_BREAKER_ENABLED = True

    def _ok_generate(*args, **kwargs):
        return "ok", 1, 1

    main._generate = _ok_generate
    try:
        main.v1_admin_circuit_breaker_open(reason="short", duration_seconds=1)
        with main.CIRCUIT_BREAKER_LOCK:
            main.CIRCUIT_BREAKER["manual_expires_at"] = main._now_ts() - 1

        out = main.chat(ChatRequest(prompt="allowed after expiry", max_new_tokens=8))
        assert out.response == "ok"
        snap = main.v1_admin_circuit_breaker_status()
        assert snap["manual_forced_open"] is False
        assert snap["state"] == "closed"
    finally:
        main._generate = prev_generate
        main.CIRCUIT_BREAKER_ENABLED = prev_enabled
        main._circuit_breaker_reset()


def test_admin_force_open_rejects_non_positive_duration():
    try:
        main.v1_admin_circuit_breaker_open(reason="bad", duration_seconds=0)
        assert False, "expected invalid_request"
    except HTTPException as exc:
        assert exc.status_code == 400
