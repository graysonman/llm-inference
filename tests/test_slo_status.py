from app import main


def setup_function():
    main.REQUEST_ID.set("rid-slo")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    with main.SLO_LOCK:
        main.SLO_EVENTS.clear()
    main.SLO_WINDOW_SECONDS = 3600
    main.SLO_ERROR_BUDGET_PCT = 1.0


def test_slo_snapshot_and_admin_endpoint():
    now = main._now_ts()
    with main.SLO_LOCK:
        main.SLO_EVENTS.extend(
            [
                {"ts": now - 5, "status_code": 200, "path": "/v1/chat"},
                {"ts": now - 4, "status_code": 503, "path": "/v1/chat"},
                {"ts": now - 3, "status_code": 200, "path": "/v1/metrics"},
                {"ts": now - 2, "status_code": 500, "path": "/v1/evaluate"},
            ]
        )

    payload = main.v1_admin_slo_status(window_seconds=60)
    assert payload["requests_total"] == 4
    assert payload["failed_requests"] == 2
    assert payload["error_rate_pct"] == 50.0
    assert payload["breached"] is True
    assert payload["request_id"] == "rid-slo"


def test_runtime_config_can_tune_slo_settings():
    out = main.v1_admin_runtime_config_update(
        {"slo_window_seconds": 120, "slo_error_budget_pct": 2.5},
        dry_run=False,
    )
    assert out["applied"] is True
    assert main.SLO_WINDOW_SECONDS == 120
    assert abs(main.SLO_ERROR_BUDGET_PCT - 2.5) < 1e-9

