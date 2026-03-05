import os
import tempfile

from fastapi import HTTPException

from app import main


def setup_function():
    main.REQUEST_ID.set("rid-runtime-profiles")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.AUDIT_LOGS.clear()
    main.RUNTIME_CONFIG_PROFILES.clear()


def test_runtime_config_profile_crud_and_apply():
    prev_timeout = main.CHAT_REQUEST_TIMEOUT_MS
    try:
        created = main.v1_admin_runtime_config_profile_upsert(
            "safe-defaults",
            payload={"chat_request_timeout_ms": 12345, "rate_limit_per_minute": 333},
            overwrite=True,
        )
        assert created["profile_name"] == "safe-defaults"
        assert created["config"]["chat_request_timeout_ms"] == 12345

        listed = main.v1_admin_runtime_config_profiles(limit=10)
        assert listed["count"] == 1
        assert listed["data"][0]["profile_name"] == "safe-defaults"

        fetched = main.v1_admin_runtime_config_profile_get("safe-defaults")
        assert fetched["config"]["rate_limit_per_minute"] == 333

        dry = main.v1_admin_runtime_config_profile_apply("safe-defaults", dry_run=True)
        assert dry["dry_run"] is True
        assert main.CHAT_REQUEST_TIMEOUT_MS != 12345

        applied = main.v1_admin_runtime_config_profile_apply("safe-defaults", dry_run=False)
        assert applied["applied"] is True
        assert main.CHAT_REQUEST_TIMEOUT_MS == 12345

        deleted = main.v1_admin_runtime_config_profile_delete("safe-defaults")
        assert deleted["deleted"] is True
        assert main.v1_admin_runtime_config_profiles(limit=10)["count"] == 0
    finally:
        main.CHAT_REQUEST_TIMEOUT_MS = prev_timeout


def test_runtime_config_profile_validation_and_conflict():
    main.v1_admin_runtime_config_profile_upsert("ops.v1", payload=None, overwrite=True)

    try:
        main.v1_admin_runtime_config_profile_upsert("ops.v1", payload=None, overwrite=False)
        assert False, "expected conflict"
    except HTTPException as exc:
        assert exc.status_code == 409

    try:
        main.v1_admin_runtime_config_profile_upsert("bad name!", payload=None, overwrite=True)
        assert False, "expected invalid name"
    except HTTPException as exc:
        assert exc.status_code == 400


def test_runtime_config_profiles_persist_roundtrip_json_backend():
    prev_enabled = main.STATE_PERSISTENCE_ENABLED
    prev_backend = main.STATE_BACKEND
    prev_path = main.STATE_FILE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "json"
        main.STATE_FILE_PATH = os.path.join(td, "state.json")

        main.v1_admin_runtime_config_profile_upsert(
            "persisted",
            payload={"chat_max_concurrent_requests": 7},
            overwrite=True,
        )

        main.RUNTIME_CONFIG_PROFILES.clear()
        main._load_state_from_disk()

        assert "persisted" in main.RUNTIME_CONFIG_PROFILES
        cfg = main.RUNTIME_CONFIG_PROFILES["persisted"]["config"]
        assert cfg["chat_max_concurrent_requests"] == 7

    main.STATE_PERSISTENCE_ENABLED = prev_enabled
    main.STATE_BACKEND = prev_backend
    main.STATE_FILE_PATH = prev_path

