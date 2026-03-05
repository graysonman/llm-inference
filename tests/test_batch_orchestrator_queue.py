import time

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-batch-queue")
    main.DATASETS.clear()
    main.BATCH_EVAL_RUNS.clear()


def _wait_for_status(run_id: str, statuses: set[str], timeout_s: float = 3.0) -> str:
    deadline = time.time() + timeout_s
    status = main.v1_get_batch_eval(run_id).status
    while status not in statuses and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run_id).status
    return status


def test_batch_orchestrator_respects_global_concurrency_and_fifo_queue():
    prev_concurrency = main.BATCH_EVAL_MAX_CONCURRENT_RUNS
    main.BATCH_EVAL_MAX_CONCURRENT_RUNS = 1
    try:
        ds = main.v1_create_dataset(
            DatasetCreateRequest(
                name="queue-ds",
                type="eval_set",
                records=[
                    {"input": "r1", "output": "o1", "_sleep_ms": "120"},
                    {"input": "r2", "output": "o2", "_sleep_ms": "120"},
                    {"input": "r3", "output": "o3", "_sleep_ms": "120"},
                ],
            )
        )

        run1 = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
        run2 = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
        run3 = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))

        # Allow dispatcher to start first run and keep others queued.
        time.sleep(0.06)
        q = main.v1_get_batch_eval_queue()
        assert q["max_concurrent_runs"] == 1
        assert q["running_count"] == 1
        assert q["queued_count"] >= 2
        queued_ids = [row["run_id"] for row in q["queued"]]
        assert run2.run_id in queued_ids
        assert run3.run_id in queued_ids

        # FIFO position should keep run2 ahead of run3.
        pos = {row["run_id"]: row["queue_position"] for row in q["queued"]}
        assert pos[run2.run_id] < pos[run3.run_id]

        assert _wait_for_status(run1.run_id, {"completed", "failed"}) == "completed"
        assert _wait_for_status(run2.run_id, {"completed", "failed"}) == "completed"
        assert _wait_for_status(run3.run_id, {"completed", "failed"}) == "completed"
    finally:
        main.BATCH_EVAL_MAX_CONCURRENT_RUNS = prev_concurrency


def test_batch_queue_endpoint_excludes_deleted_by_default():
    prev_concurrency = main.BATCH_EVAL_MAX_CONCURRENT_RUNS
    main.BATCH_EVAL_MAX_CONCURRENT_RUNS = 1
    try:
        ds = main.v1_create_dataset(
            DatasetCreateRequest(name="queue-del", type="eval_set", records=[{"input": "a", "output": "b", "_sleep_ms": "100"}])
        )
        run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))
        time.sleep(0.03)
        main.v1_delete_batch_eval(run.run_id)

        hidden = main.v1_get_batch_eval_queue()
        all_ids_hidden = [row["run_id"] for row in hidden["running"]] + [row["run_id"] for row in hidden["queued"]]
        assert run.run_id not in all_ids_hidden

        visible = main.v1_get_batch_eval_queue(include_deleted=True)
        all_ids_visible = [row["run_id"] for row in visible["running"]] + [row["run_id"] for row in visible["queued"]]
        # Deleted runs may already have completed; if still non-terminal it should appear when include_deleted=True.
        if run.run_id in main.BATCH_EVAL_RUNS and main.BATCH_EVAL_RUNS[run.run_id].get("status") in {"running", "queued"}:
            assert run.run_id in all_ids_visible
    finally:
        main.BATCH_EVAL_MAX_CONCURRENT_RUNS = prev_concurrency

