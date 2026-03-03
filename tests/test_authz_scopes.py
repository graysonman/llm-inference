import pytest
from fastapi import HTTPException

from app import main


def test_build_api_key_registry_supports_role_and_scope_overrides():
    registry = main._build_api_key_registry(
        default_key="default-key",
        default_role="admin",
        raw_json='{"viewer-key":"viewer","writer-key":{"role":"analyst","scopes":["datasets:write"]}}',
    )

    assert registry["default-key"]["role"] == "admin"
    assert registry["viewer-key"]["role"] == "viewer"
    assert "metrics:read" in registry["viewer-key"]["scopes"]
    assert registry["writer-key"]["scopes"] == ["datasets:write"]


def test_scope_allowed_supports_exact_and_wildcard():
    assert main._scope_allowed(["datasets:read"], "datasets:read")
    assert main._scope_allowed(["datasets:*"], "datasets:write")
    assert main._scope_allowed(["*"], "batch:write")
    assert not main._scope_allowed(["datasets:read"], "datasets:write")


def test_require_auth_for_scopes_rejects_missing_scope():
    previous_registry = dict(main.API_KEY_REGISTRY)
    main.API_KEY_REGISTRY = {
        "viewer-key": {"role": "viewer", "scopes": ["datasets:read"]},
    }

    try:
        with pytest.raises(HTTPException) as exc:
            main._require_auth_for_scopes(["datasets:write"], x_api_key="viewer-key", authorization=None)
        assert exc.value.status_code == 403
        detail = exc.value.detail
        assert detail["code"] == "forbidden"
        assert "datasets:write" in detail["details"]["missing_scopes"]
    finally:
        main.API_KEY_REGISTRY = previous_registry
