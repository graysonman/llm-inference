import asyncio

from app import main


def setup_function():
    main.REQUEST_ID.set("rid-stream")
    main.BATCH_EVAL_RUNS.clear()


async def _collect_stream_body(response):
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


def test_stream_endpoint_emits_events_and_done_for_completed_run():
    run_id = "be_stream_done"
    main.BATCH_EVAL_RUNS[run_id] = {
        "run_id": run_id,
        "batch_eval_id": run_id,
        "dataset_id": "ds_x",
        "status": "completed",
        "deleted_at": None,
        "events": [
            {"ts": 100, "event_type": "queued", "details": {}},
            {"ts": 101, "event_type": "running", "details": {}},
            {"ts": 102, "event_type": "completed", "details": {}},
        ],
    }

    response = asyncio.run(
        main.v1_stream_batch_eval_events(run_id=run_id, poll_interval_ms=10, max_seconds=1)
    )
    body = asyncio.run(_collect_stream_body(response))

    assert "event: meta" in body
    assert "event: batch_event" in body
    assert "\"event_type\":\"completed\"" in body
    assert "event: done" in body


def test_stream_endpoint_can_filter_event_type():
    run_id = "be_stream_filter"
    main.BATCH_EVAL_RUNS[run_id] = {
        "run_id": run_id,
        "batch_eval_id": run_id,
        "dataset_id": "ds_y",
        "status": "completed",
        "deleted_at": None,
        "events": [
            {"ts": 200, "event_type": "queued", "details": {}},
            {"ts": 201, "event_type": "progress", "details": {"completed": 1}},
            {"ts": 202, "event_type": "completed", "details": {}},
        ],
    }

    response = asyncio.run(
        main.v1_stream_batch_eval_events(
            run_id=run_id,
            event_type="progress",
            poll_interval_ms=10,
            max_seconds=1,
        )
    )
    body = asyncio.run(_collect_stream_body(response))

    assert "\"event_type\":\"progress\"" in body
    assert "\"event_type\":\"queued\"" not in body
