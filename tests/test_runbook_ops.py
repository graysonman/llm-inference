from fastapi import HTTPException

from app import main
import os
import tempfile


def setup_function():
    main.REQUEST_ID.set("rid-runbook")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.RUNBOOK_RUNS.clear()
    main.RUNBOOK_TEMPLATES.clear()
    main.AUDIT_LOGS.clear()


def test_runbook_create_step_update_complete_and_abort():
    created = main.v1_admin_runbook_create(
        {
            "name": "pre-release checklist",
            "environment": "staging",
            "steps": ["Smoke tests", "Readiness checks", "Rollback verification"],
        }
    )
    runbook = created["data"]
    rid = runbook["runbook_id"]
    assert runbook["status"] == "in_progress"
    assert len(runbook["steps"]) == 3

    main.v1_admin_runbook_step_update(rid, "step_1", {"status": "completed", "note": "ok"})
    main.v1_admin_runbook_step_update(rid, "step_2", {"status": "completed"})
    main.v1_admin_runbook_step_update(rid, "step_3", {"status": "completed"})
    completed = main.v1_admin_runbook_complete(rid)
    assert completed["data"]["status"] == "completed"

    # Create another and abort path.
    other = main.v1_admin_runbook_create(
        {"name": "hotfix checklist", "environment": "prod", "steps": ["Mitigation", "Verify"]}
    )["data"]["runbook_id"]
    aborted = main.v1_admin_runbook_abort(other, reason="operator stop")
    assert aborted["data"]["status"] == "aborted"

    listing = main.v1_admin_runbooks(limit=10)
    assert listing["count"] >= 2
    actions = [row.get("action") for row in main.AUDIT_LOGS]
    assert "runbook.create" in actions
    assert "runbook.step_update" in actions
    assert "runbook.complete" in actions
    assert "runbook.abort" in actions


def test_runbook_conflict_and_validation_cases():
    created = main.v1_admin_runbook_create({"name": "rb", "steps": ["a", "b"]})["data"]
    rid = created["runbook_id"]

    try:
        main.v1_admin_runbook_complete(rid)
        assert False, "expected conflict when steps incomplete"
    except HTTPException as exc:
        assert exc.status_code == 409

    try:
        main.v1_admin_runbook_step_update(rid, "step_99", {"status": "completed"})
        assert False, "expected missing step"
    except HTTPException as exc:
        assert exc.status_code == 404

    try:
        main.v1_admin_runbook_create({"name": "", "steps": ["a"]})
        assert False, "expected invalid name"
    except HTTPException as exc:
        assert exc.status_code == 400


def test_runbook_persistence_roundtrip_json():
    prev_enabled = main.STATE_PERSISTENCE_ENABLED
    prev_backend = main.STATE_BACKEND
    prev_path = main.STATE_FILE_PATH
    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "json"
        main.STATE_FILE_PATH = os.path.join(td, "state.json")

        runbook_id = main.v1_admin_runbook_create({"name": "persist-rb", "steps": ["s1"]})["data"]["runbook_id"]
        main.RUNBOOK_RUNS.clear()
        main._load_state_from_disk()
        assert runbook_id in main.RUNBOOK_RUNS

    main.STATE_PERSISTENCE_ENABLED = prev_enabled
    main.STATE_BACKEND = prev_backend
    main.STATE_FILE_PATH = prev_path


def test_runbook_template_crud_and_create_from_template():
    upserted = main.v1_admin_runbook_template_upsert(
        "release.v1",
        payload={"name": "Release v1", "environment": "staging", "steps": ["Freeze", "Deploy", "Verify"]},
        overwrite=True,
    )
    assert upserted["data"]["template_id"] == "release.v1"

    listed = main.v1_admin_runbook_templates(limit=10)
    assert listed["count"] == 1

    fetched = main.v1_admin_runbook_template_get("release.v1")
    assert fetched["data"]["name"] == "Release v1"

    rb = main.v1_admin_runbook_create_from_template("release.v1")
    assert rb["data"]["template_id"] == "release.v1"
    assert len(rb["data"]["steps"]) == 3

    deleted = main.v1_admin_runbook_template_delete("release.v1")
    assert deleted["deleted"] is True

    actions = [row.get("action") for row in main.AUDIT_LOGS]
    assert "runbook_template.upsert" in actions
    assert "runbook_template.delete" in actions
    assert "runbook.create_from_template" in actions
