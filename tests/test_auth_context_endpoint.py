from app import main


def setup_function():
    main.REQUEST_ID.set("rid-auth-context")
    main.AUTH_CONTEXT.set(None)


def test_v1_auth_context_uses_authenticated_context():
    main.AUTH_CONTEXT.set(
        {
            "role": "admin",
            "scopes": ["*"],
            "api_key": "admin-key",
        }
    )

    payload = main.v1_auth_context()

    assert payload["request_id"] == "rid-auth-context"
    assert payload["data"]["role"] == "admin"
    assert payload["data"]["is_admin"] is True
    assert payload["data"]["scope_count"] == 1
    assert payload["data"]["scopes"] == ["*"]
    assert payload["data"]["capabilities"]["admin.access"] is True
    assert payload["data"]["capabilities"]["chat.invoke"] is True
    assert payload["data"]["capability_count"] >= 1
    assert payload["data"]["key_fingerprint"].startswith("sha256:")


def test_v1_auth_context_defaults_when_context_missing():
    payload = main.v1_auth_context()

    assert payload["data"]["role"] == "viewer"
    assert payload["data"]["is_admin"] is False
    assert payload["data"]["scope_count"] == 0
    assert payload["data"]["scopes"] == []
    assert payload["data"]["capabilities"]["admin.access"] is False
    assert payload["data"]["capability_count"] == 0
    assert payload["data"]["key_fingerprint"] == "anonymous"


def test_v1_auth_capabilities_contains_catalog_and_effective_flags():
    main.AUTH_CONTEXT.set(
        {
            "role": "viewer",
            "scopes": ["datasets:read", "metrics:read"],
            "api_key": "viewer-key",
        }
    )

    payload = main.v1_auth_capabilities()

    assert payload["request_id"] == "rid-auth-context"
    assert payload["count"] >= 1
    rows = payload["data"]
    by_cap = {row["capability"]: row for row in rows}
    assert by_cap["datasets.read"]["allowed"] is True
    assert by_cap["metrics.read"]["allowed"] is True
    assert by_cap["datasets.write"]["allowed"] is False
    assert by_cap["admin.access"]["allowed"] is False
    assert isinstance(by_cap["chat.invoke"]["description"], str)
