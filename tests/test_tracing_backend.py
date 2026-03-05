from app import main


def setup_function():
    main.REQUEST_ID.set("rid-tracing")


def test_tracing_status_endpoint_shape():
    payload = main.v1_tracing_status()
    assert payload["request_id"] == "rid-tracing"
    assert "enabled" in payload
    assert "active" in payload
    assert "otel_available" in payload


def test_admin_tracing_probe_returns_payload_when_inactive():
    prev_active = main.TRACING_ACTIVE
    prev_tracer = main.TRACER
    try:
        main.TRACING_ACTIVE = False
        main.TRACER = None
        payload = main.v1_admin_tracing_probe(name="smoke")
        assert payload["request_id"] == "rid-tracing"
        assert payload["emitted"] is False
        assert payload["probe_name"] == "smoke"
    finally:
        main.TRACING_ACTIVE = prev_active
        main.TRACER = prev_tracer


def test_init_tracing_backend_fallback_when_deps_missing():
    prev_enabled = main.TRACING_ENABLED
    prev_otel = main.OTEL_AVAILABLE
    prev_endpoint = main.TRACING_OTLP_ENDPOINT
    try:
        main.TRACING_ENABLED = True
        main.OTEL_AVAILABLE = False
        main.TRACING_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"
        main._init_tracing_backend()
        status = main._tracing_backend_snapshot()
        assert status["active"] is False
        assert status["reason"] is not None
    finally:
        main.TRACING_ENABLED = prev_enabled
        main.OTEL_AVAILABLE = prev_otel
        main.TRACING_OTLP_ENDPOINT = prev_endpoint
        main._init_tracing_backend()
