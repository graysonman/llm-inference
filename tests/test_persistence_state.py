import json
import os
import tempfile
import time

from app import main
from app.schemas import BatchEvalCreateRequest, DatasetCreateRequest, RagQueryRequest


def setup_function():
    main.DATASETS.clear()
    main.EVAL_RUNS.clear()
    main.BATCH_EVAL_RUNS.clear()
    main.RAG_INDEXES.clear()


def test_state_save_and_load_round_trip():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_path = main.STATE_FILE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "json"
        main.STATE_FILE_PATH = os.path.join(td, "state.json")

        main.DATASETS["ds_1"] = {"dataset_id": "ds_1", "name": "a", "records": []}
        main.EVAL_RUNS["eval_1"] = {"run_id": "eval_1", "latency_ms": 12}
        main.BATCH_EVAL_RUNS["be_1"] = {"run_id": "be_1", "status": "completed", "progress": {"total": 1, "completed": 1, "failed": 0}}
        main.RAG_INDEXES["idx_1"] = {"index_id": "idx_1", "status": "ready"}

        main._save_state_to_disk()

        main.DATASETS.clear()
        main.EVAL_RUNS.clear()
        main.BATCH_EVAL_RUNS.clear()
        main.RAG_INDEXES.clear()

        main._load_state_from_disk()

        assert "ds_1" in main.DATASETS
        assert "eval_1" in main.EVAL_RUNS
        assert main.BATCH_EVAL_RUNS["be_1"]["status"] == "completed"
        assert "idx_1" in main.RAG_INDEXES

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_FILE_PATH = previous_path


def test_load_marks_inflight_batch_runs_failed():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_path = main.STATE_FILE_PATH

    with tempfile.TemporaryDirectory() as td:
        state_path = os.path.join(td, "state.json")
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "json"
        main.STATE_FILE_PATH = state_path

        payload = {
            "datasets": {},
            "eval_runs": {},
            "batch_eval_runs": {
                "be_2": {
                    "run_id": "be_2",
                    "status": "running",
                    "summary": {"mean_scores": {}, "total_items": 2, "failed_items": 0},
                }
            },
            "rag_indexes": {},
        }
        with open(state_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload))

        main._load_state_from_disk()

        assert main.BATCH_EVAL_RUNS["be_2"]["status"] == "failed"
        assert "interrupted by service restart" in main.BATCH_EVAL_RUNS["be_2"]["summary"]["error"]

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_FILE_PATH = previous_path


def test_state_sqlite_round_trip():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_sqlite_path = main.STATE_SQLITE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "sqlite"
        main.STATE_SQLITE_PATH = os.path.join(td, "state.db")

        main.DATASETS["ds_sql"] = {"dataset_id": "ds_sql", "name": "sql", "records": []}
        main.EVAL_RUNS["eval_sql"] = {"run_id": "eval_sql", "latency_ms": 7}
        main.BATCH_EVAL_RUNS["be_sql"] = {"run_id": "be_sql", "status": "completed", "progress": {"total": 1, "completed": 1, "failed": 0}}
        main.RAG_INDEXES["idx_sql"] = {"index_id": "idx_sql", "status": "ready"}

        main._save_state_to_disk()

        main.DATASETS.clear()
        main.EVAL_RUNS.clear()
        main.BATCH_EVAL_RUNS.clear()
        main.RAG_INDEXES.clear()

        main._load_state_from_disk()

        assert "ds_sql" in main.DATASETS
        assert "eval_sql" in main.EVAL_RUNS
        assert main.BATCH_EVAL_RUNS["be_sql"]["status"] == "completed"
        assert "idx_sql" in main.RAG_INDEXES

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_SQLITE_PATH = previous_sqlite_path


def test_sqlite_dataset_endpoints_can_read_without_memory_state():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_sqlite_path = main.STATE_SQLITE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "sqlite"
        main.STATE_SQLITE_PATH = os.path.join(td, "state.db")
        main.REQUEST_ID.set("rid-sqlite")

        created = main.v1_create_dataset(
            DatasetCreateRequest(name="sqlite-ds", type="eval_set", records=[{"input": "x", "output": "y"}])
        )

        main.DATASETS.clear()

        fetched = main.v1_get_dataset(created.dataset_id)
        listed = main.v1_list_datasets(limit=10)

        assert fetched.dataset_id == created.dataset_id
        assert any(item.dataset_id == created.dataset_id for item in listed.data)

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_SQLITE_PATH = previous_sqlite_path


def test_sqlite_eval_batch_rag_endpoints_can_read_without_memory_state():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_sqlite_path = main.STATE_SQLITE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "sqlite"
        main.STATE_SQLITE_PATH = os.path.join(td, "state.db")
        main.REQUEST_ID.set("rid-sqlite-read")

        # Seed one eval run directly for deterministic retrieval test.
        eval_run_id = "eval_sqlite_read"
        now = main._now_ts()
        main.EVAL_RUNS[eval_run_id] = {
            "run_id": eval_run_id,
            "prompt": "p",
            "response": "r",
            "criteria": ["overall"],
            "scores": [{"criterion": "overall", "score": 8, "rationale": "ok"}],
            "model": "test-model",
            "latency_ms": 1,
            "created_at": now,
        }

        created = main.v1_create_dataset(
            DatasetCreateRequest(name="sqlite-rag-ds", type="rag_corpus", records=[{"text": "alpha"}])
        )
        run = main.v1_create_batch_eval(
            BatchEvalCreateRequest(dataset_id=created.dataset_id, criteria=["overall"])
        )

        deadline = time.time() + 2.0
        status = main.v1_get_batch_eval(run.run_id).status
        while status != "completed" and time.time() < deadline:
            time.sleep(0.02)
            status = main.v1_get_batch_eval(run.run_id).status

        main._save_state_to_disk()

        main.EVAL_RUNS.clear()
        main.BATCH_EVAL_RUNS.clear()
        main.RAG_INDEXES.clear()
        main.DATASETS.clear()

        eval_summary = main.v1_get_eval_run(eval_run_id)
        batch_status = main.v1_get_batch_eval(run.run_id)
        batch_result = main.v1_get_batch_eval_result(run.run_id)
        batch_failures = main.v1_get_batch_eval_failures(run.run_id, limit=10)
        batch_distribution = main.v1_get_batch_eval_distribution(run.run_id, criterion="overall")
        rag_index = main.v1_get_rag_index(created.dataset_id)

        assert eval_summary.run_id == eval_run_id
        assert batch_status.run_id == run.run_id
        assert batch_result.batch_eval_id == run.run_id
        assert batch_failures.batch_eval_id == run.run_id
        assert batch_distribution.batch_eval_id == run.run_id
        assert rag_index.index_id == created.dataset_id

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_SQLITE_PATH = previous_sqlite_path


def test_sqlite_metrics_counts_and_queue_depth_without_memory_state():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_sqlite_path = main.STATE_SQLITE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "sqlite"
        main.STATE_SQLITE_PATH = os.path.join(td, "state.db")
        main.REQUEST_ID.set("rid-metrics-sql")

        created = main.v1_create_dataset(
            DatasetCreateRequest(name="metrics-ds", type="eval_set", records=[{"input": "a", "output": "b"}])
        )
        run = main.v1_create_batch_eval(
            BatchEvalCreateRequest(dataset_id=created.dataset_id, criteria=["overall"])
        )

        deadline = time.time() + 2.0
        status = main.v1_get_batch_eval(run.run_id).status
        while status != "completed" and time.time() < deadline:
            time.sleep(0.02)
            status = main.v1_get_batch_eval(run.run_id).status

        main._save_state_to_disk()
        main.DATASETS.clear()
        main.EVAL_RUNS.clear()
        main.BATCH_EVAL_RUNS.clear()
        main.RAG_INDEXES.clear()

        metrics_json = main.v1_metrics(format="json")
        dashboard = main.metrics_dashboard(window="15m")
        metrics_text = main.v1_metrics(format="prometheus").body.decode("utf-8")

        assert metrics_json["datasets"]["count"] >= 1
        assert metrics_json["batch_eval_runs"]["count"] >= 1
        assert dashboard["kpis"]["datasets"] >= 1
        assert dashboard["kpis"]["batch_runs"] >= 1
        assert "datasets_total " in metrics_text
        assert "batch_eval_runs_total " in metrics_text

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_SQLITE_PATH = previous_sqlite_path


def test_sqlite_cold_memory_write_paths_batch_and_rag_query():
    previous_enabled = main.STATE_PERSISTENCE_ENABLED
    previous_backend = main.STATE_BACKEND
    previous_sqlite_path = main.STATE_SQLITE_PATH

    with tempfile.TemporaryDirectory() as td:
        main.STATE_PERSISTENCE_ENABLED = True
        main.STATE_BACKEND = "sqlite"
        main.STATE_SQLITE_PATH = os.path.join(td, "state.db")
        main.REQUEST_ID.set("rid-sqlite-cold-write")

        created = main.v1_create_dataset(
            DatasetCreateRequest(name="cold-ds", type="rag_corpus", records=[{"text": "policy alpha"}])
        )

        main.DATASETS.clear()
        main.RAG_INDEXES.clear()

        run = main.v1_create_batch_eval(
            BatchEvalCreateRequest(dataset_id=created.dataset_id, criteria=["overall"])
        )
        rag = main.v1_rag_query(
            RagQueryRequest(query="what is policy?", dataset_id=created.dataset_id, top_k=3)
        )
        deadline = time.time() + 2.0
        status = main.v1_get_batch_eval(run.run_id).status
        while status != "completed" and time.time() < deadline:
            time.sleep(0.02)
            status = main.v1_get_batch_eval(run.run_id).status

        assert run.run_id.startswith("be_")
        assert rag.response
        assert rag.retrieved_chunks

    main.STATE_PERSISTENCE_ENABLED = previous_enabled
    main.STATE_BACKEND = previous_backend
    main.STATE_SQLITE_PATH = previous_sqlite_path
