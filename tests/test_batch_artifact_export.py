import json
import time

from fastapi import HTTPException

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-batch-artifacts")
    main.AUTH_CONTEXT.set({"role": "admin", "api_key": "admin-key"})
    main.DATASETS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.AUDIT_LOGS.clear()


def _wait_for_run_completion(run_id: str, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    status = main.v1_get_batch_eval(run_id).status
    while status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run_id).status


def test_batch_artifact_export_json_shape():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(
            name="artifacts-json",
            type="eval_set",
            records=[{"input": "a", "output": "b"}, {"input": "c", "output": "d"}],
        )
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall", "accuracy"]))
    _wait_for_run_completion(run.run_id)

    payload = main.v1_export_batch_eval_artifacts(run.run_id, format="json")
    assert payload["run_id"] == run.run_id
    assert payload["counts"]["record_scores"] == 2
    assert "overall" in payload["artifacts"]["distribution_by_criterion"]
    assert payload["filename"].endswith(".json")

    actions = [row.get("action") for row in main.AUDIT_LOGS]
    assert "batch_eval.export" in actions


def test_batch_artifact_export_csv_and_jsonl():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(
            name="artifacts-text",
            type="eval_set",
            records=[{"input": "x", "output": "y"}, {"input": "one", "output": "two"}],
        )
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
    _wait_for_run_completion(run.run_id)

    csv_response = main.v1_export_batch_eval_artifacts(run.run_id, format="csv", include_records=False)
    csv_text = csv_response.body.decode("utf-8")
    header = csv_text.splitlines()[0]
    assert "run_id" in header
    assert "score_overall" in header
    assert "record_json" not in header
    assert csv_response.headers.get("content-disposition", "").endswith(".csv\"")

    jsonl_response = main.v1_export_batch_eval_artifacts(run.run_id, format="jsonl", include_records=True)
    jsonl_lines = [ln for ln in jsonl_response.body.decode("utf-8").splitlines() if ln.strip()]
    first_row = json.loads(jsonl_lines[0])
    assert first_row["run_id"] == run.run_id
    assert "record" in first_row
    assert jsonl_response.headers.get("content-disposition", "").endswith(".jsonl\"")


def test_batch_artifact_export_deleted_visibility_and_invalid_format():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(name="artifacts-del", type="eval_set", records=[{"input": "a", "output": "b"}])
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
    _wait_for_run_completion(run.run_id)
    main.v1_delete_batch_eval(run.run_id)

    try:
        main.v1_export_batch_eval_artifacts(run.run_id, format="json")
        assert False, "expected 404 for deleted run without include_deleted"
    except HTTPException as exc:
        assert exc.status_code == 404

    visible = main.v1_export_batch_eval_artifacts(run.run_id, format="json", include_deleted=True)
    assert visible["run_id"] == run.run_id

    try:
        main.v1_export_batch_eval_artifacts(run.run_id, format="xml", include_deleted=True)
        assert False, "expected 400 for invalid export format"
    except HTTPException as exc:
        assert exc.status_code == 400

