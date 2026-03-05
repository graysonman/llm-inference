from fastapi.responses import PlainTextResponse

from app.main import (
    REQUEST_ID,
    METRICS,
    RATE_LIMIT_BUCKETS,
    v1_health,
    v1_metrics,
)


def setup_function():
    REQUEST_ID.set("rid-observe")
    METRICS["chat_requests"] = 0
    METRICS["chat_cache_hits"] = 0
    METRICS["chat_cache_misses"] = 0
    METRICS["chat_backpressure_rejections"] = 0
    METRICS["chat_timeouts"] = 0
    METRICS["maintenance_rejections"] = 0
    RATE_LIMIT_BUCKETS.clear()


def test_v1_health_shape():
    payload = v1_health()
    assert payload["status"] == "ok"
    assert "model" in payload
    assert "device" in payload
    assert payload["request_id"] == "rid-observe"


def test_v1_metrics_text_format():
    METRICS["chat_requests"] = 7
    METRICS["chat_cache_hits"] = 3
    METRICS["chat_cache_misses"] = 4
    METRICS["chat_backpressure_rejections"] = 2
    METRICS["chat_timeouts"] = 1
    METRICS["maintenance_rejections"] = 5

    response = v1_metrics(format="prometheus")
    assert isinstance(response, PlainTextResponse)
    body = response.body.decode("utf-8")
    assert "http_requests_total 7" in body
    assert "cache_hits_total 3" in body
    assert "cache_misses_total 4" in body
    assert "rate_limit_rejections_total" in body
    assert "chat_backpressure_rejections_total 2" in body
    assert "chat_timeouts_total 1" in body
    assert "maintenance_rejections_total 5" in body
    assert "maintenance_mode_active " in body
    assert "slo_error_rate_pct " in body
    assert "slo_error_budget_breached " in body
    assert "runbook_templates_total " in body
