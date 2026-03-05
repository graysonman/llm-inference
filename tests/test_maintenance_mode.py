from app import main


def setup_function():
    main.REQUEST_ID.set("rid-maint")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.AUDIT_LOGS.clear()
    main.METRICS["maintenance_rejections"] = 0
    main._maintenance_disable()


def test_maintenance_enable_disable_and_status():
    enabled = main.v1_admin_maintenance_enable(reason="deploy", duration_seconds=120, read_only=True)
    assert enabled["active"] is True
    assert enabled["read_only"] is True
    assert enabled["reason"] == "deploy"

    status = main.v1_admin_maintenance_status()
    assert status["active"] is True
    assert status["read_only"] is True

    disabled = main.v1_admin_maintenance_disable()
    assert disabled["active"] is False

    actions = [row.get("action") for row in main.AUDIT_LOGS]
    assert "maintenance.enable" in actions
    assert "maintenance.disable" in actions


def test_maintenance_gating_logic_read_only_and_full():
    main._maintenance_enable(reason="ro", duration_seconds=120, read_only=True)
    blocked_get, _ = main._maintenance_should_block_request(path="/v1/chat", method="GET", is_admin=False)
    blocked_post, details = main._maintenance_should_block_request(path="/v1/chat", method="POST", is_admin=False)
    assert blocked_get is False
    assert blocked_post is True
    assert details["read_only"] is True

    main._maintenance_enable(reason="full", duration_seconds=120, read_only=False)
    blocked_any, details2 = main._maintenance_should_block_request(path="/v1/metrics", method="GET", is_admin=False)
    assert blocked_any is True
    assert details2["reason"] == "full"

    admin_blocked, _ = main._maintenance_should_block_request(path="/v1/chat", method="POST", is_admin=True)
    assert admin_blocked is False


def test_maintenance_expiry_auto_disables():
    main._maintenance_enable(reason="short", duration_seconds=1, read_only=False)
    with main.MAINTENANCE_LOCK:
        main.MAINTENANCE_STATE["expires_at"] = main._now_ts() - 1
    assert main._maintenance_is_active() is False
    snap = main._maintenance_snapshot()
    assert snap["active"] is False

