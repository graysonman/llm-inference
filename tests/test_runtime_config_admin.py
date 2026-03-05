from fastapi import HTTPException

from app import main


def setup_function():
    main.REQUEST_ID.set("rid-runtime-config")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.AUDIT_LOGS.clear()


def test_runtime_config_get_and_dry_run_update():
    payload = main.v1_admin_runtime_config()
    assert "rate_limit_per_minute" in payload
    assert payload["request_id"] == "rid-runtime-config"

    out = main.v1_admin_runtime_config_update({"chat_request_timeout_ms": 12345}, dry_run=True)
    assert out["dry_run"] is True
    assert out["applied"] is False
    assert out["after"]["chat_request_timeout_ms"] == 12345
    assert main.CHAT_REQUEST_TIMEOUT_MS != 12345


def test_runtime_config_apply_update_and_restore():
    prev = {
        "rate_limit_per_minute": main.RATE_LIMIT_PER_MINUTE,
        "chat_max_concurrent_requests": main.CHAT_MAX_CONCURRENT_REQUESTS,
        "chat_request_timeout_ms": main.CHAT_REQUEST_TIMEOUT_MS,
        "batch_eval_max_concurrent_runs": main.BATCH_EVAL_MAX_CONCURRENT_RUNS,
    }
    try:
        out = main.v1_admin_runtime_config_update(
            {
                "rate_limit_per_minute": 222,
                "chat_max_concurrent_requests": 3,
                "chat_request_timeout_ms": 7777,
                "batch_eval_max_concurrent_runs": 4,
                "slo_window_seconds": 120,
                "slo_error_budget_pct": 2.0,
            },
            dry_run=False,
        )
        assert out["applied"] is True
        assert main.RATE_LIMIT_PER_MINUTE == 222
        assert main.CHAT_MAX_CONCURRENT_REQUESTS == 3
        assert main.CHAT_REQUEST_TIMEOUT_MS == 7777
        assert main.BATCH_EVAL_MAX_CONCURRENT_RUNS == 4
        assert main.SLO_WINDOW_SECONDS == 120
        assert abs(main.SLO_ERROR_BUDGET_PCT - 2.0) < 1e-9
        actions = [row.get("action") for row in main.AUDIT_LOGS]
        assert "runtime_config.update" in actions
    finally:
        main.RATE_LIMIT_PER_MINUTE = prev["rate_limit_per_minute"]
        main.CHAT_MAX_CONCURRENT_REQUESTS = prev["chat_max_concurrent_requests"]
        main.CHAT_REQUEST_TIMEOUT_MS = prev["chat_request_timeout_ms"]
        main.BATCH_EVAL_MAX_CONCURRENT_RUNS = prev["batch_eval_max_concurrent_runs"]


def test_runtime_config_update_rejects_invalid_payload():
    try:
        main.v1_admin_runtime_config_update({"unknown_key": 1}, dry_run=False)
        assert False, "expected invalid_request for unknown key"
    except HTTPException as exc:
        assert exc.status_code == 400

    try:
        main.v1_admin_runtime_config_update({"chat_request_timeout_ms": "bad"}, dry_run=False)
        assert False, "expected invalid_request for non-int timeout"
    except HTTPException as exc:
        assert exc.status_code == 400

    try:
        main.v1_admin_runtime_config_update({"chat_request_timeout_ms": 10}, dry_run=False)
        assert False, "expected invalid_request for out-of-range timeout"
    except HTTPException as exc:
        assert exc.status_code == 400

    try:
        main.v1_admin_runtime_config_update({"slo_error_budget_pct": 0}, dry_run=False)
        assert False, "expected invalid_request for out-of-range slo budget"
    except HTTPException as exc:
        assert exc.status_code == 400
