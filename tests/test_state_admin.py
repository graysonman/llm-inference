from app import main
from app.schemas import DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-state-admin")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.DATASETS.clear()
    main.EVAL_RUNS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.RAG_INDEXES.clear()
    main.AUDIT_LOGS.clear()
    main.RUNTIME_CONFIG_PROFILES.clear()
    main.RUNBOOK_RUNS.clear()
    main.RUNBOOK_TEMPLATES.clear()
    main._maintenance_disable()
    with main.SLO_LOCK:
        main.SLO_EVENTS.clear()
        main.SLO_INCIDENTS.clear()
        main.SLO_STATE["breached"] = False
        main.SLO_STATE["current_incident_id"] = None


def test_state_export_shape():
    main.v1_create_dataset(DatasetCreateRequest(name="ex", type="eval_set", records=[{"input": "a", "output": "b"}]))
    payload = main.v1_admin_state_export()
    assert payload["schema_version"] == 1
    assert "data" in payload
    assert "datasets" in payload["data"]
    assert payload["request_id"] == "rid-state-admin"


def test_state_import_dry_run_does_not_mutate():
    main.DATASETS["ds_live"] = {"dataset_id": "ds_live", "name": "live", "type": "eval_set", "status": "ready", "record_count": 1}
    incoming = {
        "data": {
            "datasets": {"ds_new": {"dataset_id": "ds_new", "name": "new", "type": "eval_set", "status": "ready", "record_count": 2}},
            "eval_runs": {},
            "batch_eval_runs": {},
            "rag_indexes": {},
            "audit_logs": [],
        }
    }
    payload = main.v1_admin_state_import(incoming, mode="replace", dry_run=True)
    assert payload["dry_run"] is True
    assert "ds_live" in main.DATASETS
    assert "ds_new" not in main.DATASETS


def test_state_import_replace_and_merge():
    main.DATASETS["ds_old"] = {"dataset_id": "ds_old", "name": "old", "type": "eval_set", "status": "ready", "record_count": 1}
    replace_payload = {
        "data": {
            "datasets": {"ds_rep": {"dataset_id": "ds_rep", "name": "rep", "type": "eval_set", "status": "ready", "record_count": 2}},
            "eval_runs": {},
            "batch_eval_runs": {},
            "rag_indexes": {},
            "audit_logs": [],
        }
    }
    out_replace = main.v1_admin_state_import(replace_payload, mode="replace", dry_run=False)
    assert out_replace["mode"] == "replace"
    assert "ds_old" not in main.DATASETS
    assert "ds_rep" in main.DATASETS

    merge_payload = {
        "data": {
            "datasets": {"ds_merge": {"dataset_id": "ds_merge", "name": "merge", "type": "eval_set", "status": "ready", "record_count": 3}},
            "eval_runs": {},
            "batch_eval_runs": {},
            "rag_indexes": {},
            "audit_logs": [],
        }
    }
    out_merge = main.v1_admin_state_import(merge_payload, mode="merge", dry_run=False)
    assert out_merge["mode"] == "merge"
    assert "ds_rep" in main.DATASETS
    assert "ds_merge" in main.DATASETS


def test_state_import_replace_applies_maintenance_state():
    payload = {
        "data": {
            "datasets": {},
            "eval_runs": {},
            "batch_eval_runs": {},
            "rag_indexes": {},
            "audit_logs": [],
            "runtime_config_profiles": {},
            "maintenance_state": {"active": True, "reason": "window", "read_only": True, "enabled_at": 1, "expires_at": None},
        }
    }
    main.v1_admin_state_import(payload, mode="replace", dry_run=False)
    status = main.v1_admin_maintenance_status()
    assert status["active"] is True
    assert status["read_only"] is True
    assert status["reason"] == "window"


def test_state_import_replace_applies_slo_incident_state():
    payload = {
        "data": {
            "datasets": {},
            "eval_runs": {},
            "batch_eval_runs": {},
            "rag_indexes": {},
            "audit_logs": [],
            "runtime_config_profiles": {},
            "maintenance_state": {},
            "slo_incidents": [
                {
                    "incident_id": "sloi_test",
                    "status": "open",
                    "opened_at": 1,
                    "updated_at": 1,
                    "resolved_at": None,
                    "peak_error_rate_pct": 5.0,
                    "latest_error_rate_pct": 5.0,
                    "error_budget_pct": 1.0,
                    "window_seconds": 3600,
                    "request_count_at_open": 10,
                    "failed_count_at_open": 1,
                }
            ],
            "slo_state": {"breached": True, "current_incident_id": "sloi_test"},
        }
    }
    main.v1_admin_state_import(payload, mode="replace", dry_run=False)
    cur = main.v1_admin_slo_incident_current()
    assert cur["data"] is not None
    assert cur["data"]["incident_id"] == "sloi_test"
