from app.main import (
    REQUEST_ID,
    DATASETS,
    BATCH_EVAL_RUNS,
    _resolve_api_key,
    _parse_dataset_upload,
    _validate_records,
    v1_create_dataset,
    v1_rag,
    metrics_dashboard,
)
from app.schemas import DatasetCreateRequest, RagQueryRequest


def setup_function():
    REQUEST_ID.set("rid-contract")
    DATASETS.clear()
    BATCH_EVAL_RUNS.clear()


def test_resolve_api_key_supports_bearer_and_x_api_key():
    assert _resolve_api_key("x-key", None) == "x-key"
    assert _resolve_api_key(None, "Bearer bearer-key") == "bearer-key"
    assert _resolve_api_key(None, "Token nope") is None


def test_v1_rag_contract_shape():
    ds = v1_create_dataset(
        DatasetCreateRequest(
            name="rag-ds",
            type="rag_corpus",
            records=[{"text": "Rotate keys in phases."}],
        )
    )

    payload = RagQueryRequest(query="How to rotate keys?", dataset_id=ds.dataset_id, top_k=3)
    result = v1_rag(payload)

    assert result.answer
    assert result.retrieval.dataset_id == ds.dataset_id
    assert result.retrieval.top_k == 3
    assert len(result.citations) == 1


def test_dataset_upload_parsing_and_validation_helpers():
    raw = b'{"text":"ok"}\n{"input":{"prompt":"hello"}}\n{}\n'
    parsed = _parse_dataset_upload("sample.jsonl", raw)
    accepted, errors = _validate_records(parsed)

    assert len(parsed) == 3
    assert len(accepted) == 2
    assert len(errors) == 1


def test_metrics_dashboard_shape():
    payload = metrics_dashboard(window="15m")
    assert payload["window"] == "15m"
    assert "kpis" in payload
    assert "cache" in payload