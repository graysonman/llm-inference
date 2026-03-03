from app import main


def setup_function():
    main.REQUEST_ID.set("rid-retention")
    main.DATASETS.clear()
    main.EVAL_RUNS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.RAG_INDEXES.clear()
    main.RETENTION_STATS["last_run_ts"] = None
    main.RETENTION_STATS["last_purged_total"] = 0
    main.RETENTION_STATS["last_error"] = None
    main.RETENTION_HISTORY.clear()


def test_purge_expired_soft_deleted_records_in_memory():
    now = 1_000
    cutoff_ttl = 100

    main.DATASETS["ds_old"] = {"dataset_id": "ds_old", "deleted_at": 850}
    main.DATASETS["ds_new"] = {"dataset_id": "ds_new", "deleted_at": 980}
    main.EVAL_RUNS["ev_old"] = {"run_id": "ev_old", "deleted_at": 880}
    main.EVAL_RUNS["ev_active"] = {"run_id": "ev_active", "deleted_at": None}
    main.BATCH_EVAL_RUNS["be_old"] = {"run_id": "be_old", "deleted_at": 890}
    main.RAG_INDEXES["idx_old"] = {"index_id": "idx_old", "deleted_at": 890}

    result = main._purge_expired_soft_deleted_records(retention_seconds=cutoff_ttl, now_ts=now)

    assert result["purged_total"] >= 3
    assert "ds_old" not in main.DATASETS
    assert "ev_old" not in main.EVAL_RUNS
    assert "be_old" not in main.BATCH_EVAL_RUNS
    assert "ds_new" in main.DATASETS
    assert "ev_active" in main.EVAL_RUNS


def test_admin_retention_sweep_endpoint_shape():
    main.DATASETS["ds_x"] = {"dataset_id": "ds_x", "deleted_at": 0}
    payload = main.v1_admin_retention_sweep()
    assert "request_id" in payload
    assert "purged_total" in payload
    assert "retention_seconds" in payload
    assert main.RETENTION_STATS["last_run_ts"] is not None
    assert main.RETENTION_STATS["last_purged_total"] == payload["purged_total"]

    metrics = main.v1_metrics(format="json")
    assert "retention" in metrics
    assert metrics["retention"]["last_run_ts"] == main.RETENTION_STATS["last_run_ts"]
    assert metrics["retention"]["last_purged_total"] == payload["purged_total"]
    assert len(metrics["retention"]["recent_runs"]) >= 1


def test_retention_history_ring_buffer_cap():
    previous_limit = main.RETENTION_HISTORY_LIMIT
    main.RETENTION_HISTORY_LIMIT = 2
    try:
        main._record_retention_run({"retention_seconds": 10, "cutoff_ts": 1, "purged_total": 1}, error=None, trigger="manual")
        main._record_retention_run({"retention_seconds": 10, "cutoff_ts": 2, "purged_total": 2}, error=None, trigger="manual")
        main._record_retention_run({"retention_seconds": 10, "cutoff_ts": 3, "purged_total": 3}, error=None, trigger="manual")
        assert len(main.RETENTION_HISTORY) == 2
        assert main.RETENTION_HISTORY[-1]["purged_total"] == 3
    finally:
        main.RETENTION_HISTORY_LIMIT = previous_limit


def test_retention_preview_is_non_mutating():
    main.DATASETS["ds_old"] = {"dataset_id": "ds_old", "deleted_at": 1}
    main.EVAL_RUNS["ev_old"] = {"run_id": "ev_old", "deleted_at": 1}

    preview = main.v1_admin_retention_preview()
    assert "candidate_total" in preview
    assert preview["candidate_total"] >= 2
    assert "ds_old" in preview["candidates"]["dataset_ids"]
    assert "ev_old" in preview["candidates"]["eval_run_ids"]

    # Preview should not mutate state.
    assert "ds_old" in main.DATASETS
    assert "ev_old" in main.EVAL_RUNS

    compact = main.v1_admin_retention_preview(include_ids=False)
    assert compact["include_ids"] is False
    assert compact["candidates"]["dataset_ids"] == []
    assert compact["candidates"]["eval_run_ids"] == []


def test_retention_override_query_windows():
    main.DATASETS["ds_old_2"] = {"dataset_id": "ds_old_2", "deleted_at": 900}
    # deterministic helper check
    preview_narrow = main._preview_expired_soft_deleted_records(retention_seconds=0, now_ts=1_000)
    preview_wide = main._preview_expired_soft_deleted_records(retention_seconds=10_000, now_ts=1_000)
    assert preview_wide["candidate_total"] <= preview_narrow["candidate_total"]

    # Endpoint accepts override and applies it.
    main.DATASETS["ds_override"] = {"dataset_id": "ds_override", "deleted_at": 1}
    payload = main.v1_admin_retention_sweep(retention_seconds=0)
    assert payload["retention_seconds"] == 0
    assert payload["purged_total"] >= 1


def test_retention_preview_candidate_id_limit():
    main.DATASETS["ds_a"] = {"dataset_id": "ds_a", "deleted_at": 1}
    main.DATASETS["ds_b"] = {"dataset_id": "ds_b", "deleted_at": 1}
    main.DATASETS["ds_c"] = {"dataset_id": "ds_c", "deleted_at": 1}

    payload = main.v1_admin_retention_preview(retention_seconds=0, include_ids=True, candidate_ids_limit=2)
    assert payload["candidate_datasets"] >= 3
    assert len(payload["candidates"]["dataset_ids"]) == 2
    assert payload["candidate_ids_truncated"] is True
