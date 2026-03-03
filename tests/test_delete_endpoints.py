import time

from fastapi import HTTPException

from app.main import (
    REQUEST_ID,
    DATASETS,
    EVAL_RUNS,
    BATCH_EVAL_RUNS,
    RAG_INDEXES,
    v1_create_dataset,
    v1_create_batch_eval,
    v1_get_batch_eval,
    v1_get_dataset,
    v1_get_eval_run,
    v1_get_rag_index,
    v1_delete_dataset,
    v1_delete_eval_run,
    v1_delete_batch_eval,
    v1_delete_rag_index,
    v1_restore_dataset,
    v1_restore_eval_run,
    v1_restore_batch_eval,
    v1_restore_rag_index,
)
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest


def setup_function():
    REQUEST_ID.set("rid-delete")
    DATASETS.clear()
    EVAL_RUNS.clear()
    BATCH_EVAL_RUNS.clear()
    RAG_INDEXES.clear()


def test_delete_dataset_and_rag_index():
    created = v1_create_dataset(
        DatasetCreateRequest(name="delete-ds", type="rag_corpus", records=[{"text": "x"}])
    )

    deleted = v1_delete_dataset(created.dataset_id)
    assert deleted["deleted"] is True

    try:
        v1_get_dataset(created.dataset_id)
        assert False, "expected dataset 404"
    except HTTPException as exc:
        assert exc.status_code == 404

    try:
        v1_get_rag_index(created.dataset_id)
        assert False, "expected rag index 404"
    except HTTPException as exc:
        assert exc.status_code == 404

    # Soft-deleted records remain retrievable only when explicitly requested.
    deleted_ds = v1_get_dataset(created.dataset_id, include_deleted=True)
    deleted_idx = v1_get_rag_index(created.dataset_id, include_deleted=True)
    assert deleted_ds.dataset_id == created.dataset_id
    assert deleted_idx.index_id == created.dataset_id

    restored = v1_restore_dataset(created.dataset_id)
    assert restored["restored"] is True
    active_ds = v1_get_dataset(created.dataset_id)
    active_idx = v1_get_rag_index(created.dataset_id)
    assert active_ds.dataset_id == created.dataset_id
    assert active_idx.index_id == created.dataset_id


def test_delete_eval_and_batch():
    EVAL_RUNS["eval_del"] = {
        "run_id": "eval_del",
        "prompt": "p",
        "response": "r",
        "criteria": ["overall"],
        "scores": [{"criterion": "overall", "score": 7, "rationale": "ok"}],
        "model": "m",
        "latency_ms": 1,
        "created_at": int(time.time()),
    }
    deleted_eval = v1_delete_eval_run("eval_del")
    assert deleted_eval["deleted"] is True
    try:
        v1_get_eval_run("eval_del")
        assert False, "expected eval 404"
    except HTTPException as exc:
        assert exc.status_code == 404
    eval_deleted = v1_get_eval_run("eval_del", include_deleted=True)
    assert eval_deleted.run_id == "eval_del"
    eval_restored = v1_restore_eval_run("eval_del")
    assert eval_restored["restored"] is True
    assert v1_get_eval_run("eval_del").run_id == "eval_del"

    ds = v1_create_dataset(
        DatasetCreateRequest(name="delete-batch", type="eval_set", records=[{"input": "a", "output": "b"}])
    )
    run = v1_create_batch_eval(BatchEvalCreateRequest(dataset_id=ds.dataset_id, criteria=["overall"]))

    deadline = time.time() + 2.0
    status = v1_get_batch_eval(run.run_id).status
    while status != "completed" and time.time() < deadline:
        time.sleep(0.02)
        status = v1_get_batch_eval(run.run_id).status

    deleted_batch = v1_delete_batch_eval(run.run_id)
    assert deleted_batch["deleted"] is True
    try:
        v1_get_batch_eval(run.run_id)
        assert False, "expected batch 404"
    except HTTPException as exc:
        assert exc.status_code == 404
    batch_deleted = v1_get_batch_eval(run.run_id, include_deleted=True)
    assert batch_deleted.run_id == run.run_id
    batch_restored = v1_restore_batch_eval(run.run_id)
    assert batch_restored["restored"] is True
    assert v1_get_batch_eval(run.run_id).run_id == run.run_id


def test_delete_rag_index_not_found():
    try:
        v1_delete_rag_index("missing")
        assert False, "expected 404"
    except HTTPException as exc:
        assert exc.status_code == 404


def test_restore_rag_index_round_trip():
    created = v1_create_dataset(
        DatasetCreateRequest(name="restore-rag", type="rag_corpus", records=[{"text": "z"}])
    )
    deleted = v1_delete_rag_index(created.dataset_id)
    assert deleted["deleted"] is True
    restored = v1_restore_rag_index(created.dataset_id)
    assert restored["restored"] is True
    assert v1_get_rag_index(created.dataset_id).index_id == created.dataset_id
