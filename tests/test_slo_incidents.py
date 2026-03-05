from app import main


def setup_function():
    main.REQUEST_ID.set("rid-slo-inc")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.SLO_WINDOW_SECONDS = 3600
    main.SLO_ERROR_BUDGET_PCT = 1.0
    with main.SLO_LOCK:
        main.SLO_EVENTS.clear()
        main.SLO_INCIDENTS.clear()
        main.SLO_STATE["breached"] = False
        main.SLO_STATE["current_incident_id"] = None


def test_slo_incident_opens_and_resolves():
    # First event breaches immediately (100% errors on 1 request)
    main._record_slo_event(503, "/v1/chat")
    incidents = main.v1_admin_slo_incidents(limit=10)
    assert incidents["count"] == 1
    first = incidents["data"][0]
    assert first["status"] == "open"

    current = main.v1_admin_slo_incident_current()
    assert current["data"] is not None
    incident_id = current["data"]["incident_id"]

    # Add enough successes to recover below 1% budget.
    for _ in range(300):
        main._record_slo_event(200, "/v1/chat")

    current_after = main.v1_admin_slo_incident_current()
    assert current_after["data"] is None
    incidents_after = main.v1_admin_slo_incidents(limit=10)
    resolved = [x for x in incidents_after["data"] if x["incident_id"] == incident_id][0]
    assert resolved["status"] == "resolved"
    assert resolved["resolved_at"] is not None


def test_slo_incident_filters_and_invalid_status():
    main._record_slo_event(503, "/v1/chat")
    for _ in range(200):
        main._record_slo_event(200, "/v1/chat")

    open_rows = main.v1_admin_slo_incidents(limit=10, status="open")
    resolved_rows = main.v1_admin_slo_incidents(limit=10, status="resolved")
    assert open_rows["count"] + resolved_rows["count"] >= 1

    try:
        main.v1_admin_slo_incidents(limit=10, status="bad")
        assert False, "expected invalid status error"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400


def test_slo_incident_detail_ack_and_notes():
    main._record_slo_event(503, "/v1/chat")
    cur = main.v1_admin_slo_incident_current()
    incident_id = cur["data"]["incident_id"]

    fetched = main.v1_admin_slo_incident_get(incident_id)
    assert fetched["data"]["incident_id"] == incident_id

    acked = main.v1_admin_slo_incident_ack(incident_id, note="triage started")
    assert acked["acknowledged"] is True

    noted = main.v1_admin_slo_incident_add_note(incident_id, note="captured dashboard screenshot")
    assert noted["noted"] is True

    fetched2 = main.v1_admin_slo_incident_get(incident_id)
    assert fetched2["data"]["acknowledged_at"] is not None
    notes = fetched2["data"].get("notes", [])
    assert any(str(n.get("type")) == "ack_note" for n in notes)
    assert any(str(n.get("type")) == "note" for n in notes)


def test_slo_incident_not_found_and_empty_note_validation():
    try:
        main.v1_admin_slo_incident_get("missing")
        assert False, "expected not_found"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404

    main._record_slo_event(503, "/v1/chat")
    incident_id = main.v1_admin_slo_incident_current()["data"]["incident_id"]
    try:
        main.v1_admin_slo_incident_add_note(incident_id, note="   ")
        assert False, "expected invalid_request"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
