import time

import pytest
from fastapi import HTTPException

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-purge")
    main.DATASETS.clear()
    main.EVAL_RUNS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.RAG_INDEXES.clear()
    main.API_KEY_REGISTRY = {"admin-key": {"role": "admin", "scopes": ["*"]}, "viewer-key": {"role": "viewer", "scopes": ["datasets:read"]}}


def test_require_admin_rejects_non_admin():
    with pytest.raises(HTTPException) as exc:
        main._require_admin(x_api_key="viewer-key", authorization=None)
    assert exc.value.status_code == 403


def test_purge_roundtrip_dataset_eval_batch_rag():
    created = main.v1_create_dataset(
        DatasetCreateRequest(name="purge-ds", type="rag_corpus", records=[{"text": "alpha"}])
    )
    main.v1_delete_dataset(created.dataset_id)
    purged_dataset = main.v1_purge_dataset(created.dataset_id)
    assert purged_dataset["purged"] is True
    with pytest.raises(HTTPException):
        main.v1_get_dataset(created.dataset_id, include_deleted=True)

    eval_run_id = "eval_purge"
    main.EVAL_RUNS[eval_run_id] = {
        "run_id": eval_run_id,
        "prompt": "p",
        "response": "r",
        "criteria": ["overall"],
        "scores": [{"criterion": "overall", "score": 5, "rationale": "ok"}],
        "model": "m",
        "latency_ms": 1,
        "created_at": int(time.time()),
        "deleted_at": None,
    }
    main.v1_delete_eval_run(eval_run_id)
    purged_eval = main.v1_purge_eval_run(eval_run_id)
    assert purged_eval["purged"] is True
    with pytest.raises(HTTPException):
        main.v1_get_eval_run(eval_run_id, include_deleted=True)

    ds2 = main.v1_create_dataset(
        DatasetCreateRequest(name="purge-batch", type="eval_set", records=[{"input": "x", "output": "y"}])
    )
    run = main.v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds2.dataset_id, criteria=["overall"]))
    deadline = time.time() + 2.0
    status = main.v1_get_batch_eval(run.run_id).status
    while status != "completed" and time.time() < deadline:
        time.sleep(0.02)
        status = main.v1_get_batch_eval(run.run_id).status

    main.v1_delete_batch_eval(run.run_id)
    purged_batch = main.v1_purge_batch_eval(run.run_id)
    assert purged_batch["purged"] is True
    with pytest.raises(HTTPException):
        main.v1_get_batch_eval(run.run_id, include_deleted=True)

    ds3 = main.v1_create_dataset(
        DatasetCreateRequest(name="purge-rag", type="rag_corpus", records=[{"text": "beta"}])
    )
    main.v1_delete_rag_index(ds3.dataset_id)
    purged_index = main.v1_purge_rag_index(ds3.dataset_id)
    assert purged_index["purged"] is True
    with pytest.raises(HTTPException):
        main.v1_get_rag_index(ds3.dataset_id, include_deleted=True)
