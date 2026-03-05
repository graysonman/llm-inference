import time

from fastapi import HTTPException

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-batch-events")
    main.DATASETS.clear()
    main.BATCH_EVAL_RUNS.clear()


def test_batch_eval_events_timeline_and_filter():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(
            name="events-ds",
            type="eval_set",
            records=[{"input": "a", "output": "b"}, {"input": "c", "output": "d"}],
        )
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))

    deadline = time.time() + 2.0
    status = main.v1_get_batch_eval(run.run_id).status
    while status != "completed" and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run.run_id).status

    events_payload = main.v1_get_batch_eval_events(run.run_id, limit=200)
    event_types = [row.get("event_type") for row in events_payload["data"]]
    assert "queued" in event_types
    assert "running" in event_types
    assert "completed" in event_types

    progress_only = main.v1_get_batch_eval_events(run.run_id, event_type="progress")
    assert progress_only["count"] >= 1
    assert all(row.get("event_type") == "progress" for row in progress_only["data"])


def test_batch_eval_events_respect_deleted_visibility():
    ds = main.v1_create_dataset(
        DatasetCreateRequest(name="events-del", type="eval_set", records=[{"input": "a", "output": "b"}])
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))

    deadline = time.time() + 2.0
    status = main.v1_get_batch_eval(run.run_id).status
    while status != "completed" and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run.run_id).status

    main.v1_delete_batch_eval(run.run_id)

    try:
        main.v1_get_batch_eval_events(run.run_id)
        assert False, "expected 404 for deleted run without include_deleted"
    except HTTPException as exc:
        assert exc.status_code == 404

    visible = main.v1_get_batch_eval_events(run.run_id, include_deleted=True)
    assert visible["count"] >= 1
