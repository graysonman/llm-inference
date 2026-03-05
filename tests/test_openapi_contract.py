from app.main import app


REQUIRED_ROUTES = {
    "/healthz": {"get"},
    "/readyz": {"get"},
    "/v1/auth/context": {"get"},
    "/v1/auth/capabilities": {"get"},
    "/v1/agent/tools": {"get"},
    "/v1/agent/run": {"post"},
    "/v1/chat": {"post"},
    "/v1/evaluate": {"post"},
    "/v1/embeddings": {"post"},
    "/v1/rag/query": {"post"},
    "/v1/rag/vector-backend": {"get"},
    "/v1/tracing/status": {"get"},
    "/v1/datasets": {"post", "get"},
    "/v1/datasets/{dataset_id}": {"get", "delete"},
    "/v1/datasets/{dataset_id}/restore": {"post"},
    "/v1/datasets/{dataset_id}/purge": {"delete"},
    "/v1/batch-evals": {"post"},
    "/v1/batch-evals/queue": {"get"},
    "/v1/batch-evals/{run_id}": {"get", "delete"},
    "/v1/batch-evals/{run_id}/cancel": {"post"},
    "/v1/batch-evals/{run_id}/events": {"get"},
    "/v1/batch-evals/{run_id}/retries": {"get"},
    "/v1/batch-evals/{run_id}/events/stream": {"get"},
    "/v1/batch-evals/{run_id}/artifacts/export": {"get"},
    "/v1/batch-evals/{run_id}/restore": {"post"},
    "/v1/batch-evals/{run_id}/purge": {"delete"},
    "/v1/evals/{run_id}": {"get", "delete"},
    "/v1/evals/{run_id}/restore": {"post"},
    "/v1/evals/{run_id}/purge": {"delete"},
    "/v1/rag/indexes/{index_id}": {"get", "delete"},
    "/v1/rag/indexes/{index_id}/restore": {"post"},
    "/v1/rag/indexes/{index_id}/purge": {"delete"},
    "/v1/admin/retention/preview": {"get"},
    "/v1/admin/rag/vector-backend": {"get"},
    "/v1/admin/tracing/status": {"get"},
    "/v1/admin/tracing/probe": {"post"},
    "/v1/admin/runbooks": {"get", "post"},
    "/v1/admin/runbooks/from-template/{template_id}": {"post"},
    "/v1/admin/runbooks/{runbook_id}": {"get"},
    "/v1/admin/runbooks/{runbook_id}/steps/{step_id}": {"post"},
    "/v1/admin/runbooks/{runbook_id}/complete": {"post"},
    "/v1/admin/runbooks/{runbook_id}/abort": {"post"},
    "/v1/admin/runbook-templates": {"get"},
    "/v1/admin/runbook-templates/{template_id}": {"get", "post", "delete"},
    "/v1/admin/maintenance": {"get"},
    "/v1/admin/maintenance/enable": {"post"},
    "/v1/admin/maintenance/disable": {"post"},
    "/v1/admin/slo/status": {"get"},
    "/v1/admin/slo/incidents": {"get"},
    "/v1/admin/slo/incidents/current": {"get"},
    "/v1/admin/slo/incidents/{incident_id}": {"get"},
    "/v1/admin/slo/incidents/{incident_id}/ack": {"post"},
    "/v1/admin/slo/incidents/{incident_id}/notes": {"post"},
    "/v1/admin/circuit-breaker": {"get"},
    "/v1/admin/circuit-breaker/open": {"post"},
    "/v1/admin/circuit-breaker/reset": {"post"},
    "/v1/admin/runtime-config": {"get", "post"},
    "/v1/admin/runtime-config/profiles": {"get"},
    "/v1/admin/runtime-config/profiles/{profile_name}": {"get", "post", "delete"},
    "/v1/admin/runtime-config/profiles/{profile_name}/apply": {"post"},
    "/v1/admin/audit-logs": {"get"},
    "/v1/admin/state/export": {"get"},
    "/v1/admin/state/import": {"post"},
    "/v1/metrics": {"get"},
}


def test_required_v1_paths_exist_in_openapi():
    schema = app.openapi()
    paths = schema["paths"]

    for path, methods in REQUIRED_ROUTES.items():
        assert path in paths, f"missing path {path}"
        actual_methods = set(paths[path].keys())
        for method in methods:
            assert method in actual_methods, f"missing {method.upper()} {path}"


def test_api_key_header_is_declared_for_protected_v1_routes():
    schema = app.openapi()
    protected_paths = [p for p in REQUIRED_ROUTES if p.startswith("/v1/")]
    for path in protected_paths:
        for method in REQUIRED_ROUTES[path]:
            operation = schema["paths"][path][method]
            params = operation.get("parameters", [])
            header_names = {p["name"] for p in params if p.get("in") == "header"}
            assert "x-api-key" in header_names, f"missing x-api-key header on {method.upper()} {path}"
