from app import main
from app.schemas import DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-audit")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.DATASETS.clear()
    main.EVAL_RUNS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.RAG_INDEXES.clear()
    main.AUDIT_LOGS.clear()


def test_admin_actions_are_audited_and_queryable():
    created = main.v1_create_dataset(
        DatasetCreateRequest(name="audit-ds", type="rag_corpus", records=[{"text": "hello"}])
    )
    main.v1_delete_dataset(created.dataset_id)
    main.v1_restore_dataset(created.dataset_id)
    main.v1_purge_rag_index(created.dataset_id)

    payload = main.v1_admin_audit_logs(limit=50)
    assert payload["count"] >= 3
    actions = [row["action"] for row in payload["data"]]
    assert "dataset.delete" in actions
    assert "dataset.restore" in actions
    assert "rag_index.purge" in actions


def test_audit_log_filters():
    main._record_audit_event(action="retention.preview", resource_type="retention", details={"x": 1})
    main._record_audit_event(action="dataset.delete", resource_type="dataset", resource_id="ds_1")

    filtered = main.v1_admin_audit_logs(limit=10, action="dataset.delete")
    assert filtered["count"] == 1
    assert filtered["data"][0]["action"] == "dataset.delete"

    filtered_resource = main.v1_admin_audit_logs(limit=10, resource_type="retention")
    assert filtered_resource["count"] == 1
    assert filtered_resource["data"][0]["resource_type"] == "retention"
