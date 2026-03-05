import time

from fastapi import HTTPException

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-batch-retries")
    main.DATASETS.clear()
    main.BATCH_EVAL_RUNS.clear()


def _wait_terminal(run_id: str, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    status = main.v1_get_batch_eval(run_id).status
    while status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run_id).status


def test_batch_retry_semantics_and_endpoint():
    prev_retries = main.BATCH_EVAL_MAX_RETRIES
    prev_backoff = main.BATCH_EVAL_RETRY_BACKOFF_MS
    main.BATCH_EVAL_MAX_RETRIES = 2
    main.BATCH_EVAL_RETRY_BACKOFF_MS = 0
    try:
        ds = main.v1_create_dataset(
            DatasetCreateRequest(
                name="retry-ds",
                type="eval_set",
                records=[
                    {"input": "transient", "output": "ok", "_transient_failures": "1"},
                    {"input": "always-fail", "output": "bad", "_always_fail": "true"},
                    {"input": "normal", "output": "ok"},
                ],
            )
        )
        run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
        _wait_terminal(run.run_id)

        result = main.v1_get_batch_eval_result(run.run_id)
        retry_stats = result.summary.get("retry_stats", {})
        assert result.status == "completed"
        assert retry_stats.get("total_retries", 0) >= 3
        assert retry_stats.get("exhausted_records", 0) >= 1

        retries_payload = main.v1_get_batch_eval_retries(run.run_id, limit=200)
        assert retries_payload["retry_stats"]["total_retries"] >= 3
        retry_indices = {row.get("record_index") for row in retries_payload["data"]}
        assert 0 in retry_indices
        assert 1 in retry_indices

        events = main.v1_get_batch_eval_events(run.run_id, limit=300)
        event_types = [row.get("event_type") for row in events["data"]]
        assert "retry" in event_types
    finally:
        main.BATCH_EVAL_MAX_RETRIES = prev_retries
        main.BATCH_EVAL_RETRY_BACKOFF_MS = prev_backoff


def test_batch_retries_deleted_visibility():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(
            name="retry-del",
            type="eval_set",
            records=[{"input": "a", "output": "b", "_transient_failures": "1"}],
        )
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
    _wait_terminal(run.run_id)
    main.v1_delete_batch_eval(run.run_id)

    try:
        main.v1_get_batch_eval_retries(run.run_id)
        assert False, "expected 404 for deleted run without include_deleted"
    except HTTPException as exc:
        assert exc.status_code == 404

    visible = main.v1_get_batch_eval_retries(run.run_id, include_deleted=True)
    assert "retry_stats" in visible
