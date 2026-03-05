import time

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-cancel")
    main.DATASETS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.AUDIT_LOGS.clear()


def test_cancel_queued_or_running_batch_run():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(
            name="cancel-ds",
            type="eval_set",
            records=[{"input": "a", "output": "b"} for _ in range(40)],
        )
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))

    # Issue cancel quickly; depending on timing this may hit queued or running.
    cancel = main.v1_cancel_batch_eval(run.run_id)
    assert cancel["cancel_requested"] is True

    deadline = time.time() + 3.0
    status = main.v1_get_batch_eval(run.run_id, include_deleted=True).status
    while status not in {"failed", "completed"} and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run.run_id, include_deleted=True).status

    final = main.v1_get_batch_eval_result(run.run_id, include_deleted=True)
    assert final.status == "failed"
    assert "error" in final.summary

    events = main.v1_get_batch_eval_events(run.run_id, include_deleted=True, limit=200)
    event_types = [e.get("event_type") for e in events["data"]]
    assert "cancel_requested" in event_types
    assert "failed" in event_types


def test_cancel_already_terminal_run_is_noop():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(
            name="cancel-noop",
            type="eval_set",
            records=[{"input": "x", "output": "y"}],
        )
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
    deadline = time.time() + 2.0
    status = main.v1_get_batch_eval(run.run_id).status
    while status != "completed" and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run.run_id).status

    cancel = main.v1_cancel_batch_eval(run.run_id)
    assert cancel["status"] in {"completed", "failed"}
