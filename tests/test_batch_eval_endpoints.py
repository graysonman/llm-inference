from app.main import (
    REQUEST_ID,
    DATASETS,
    BATCH_EVAL_RUNS,
    v1_create_dataset,
    v1_create_batch_eval,
    v1_get_batch_eval_result,
    v1_get_batch_eval_failures,
    v1_get_batch_eval_distribution,
)
from app.schemas import DatasetCreateRequest, BatchEvalCreateRequest


def setup_function():
    REQUEST_ID.set("rid-test")
    DATASETS.clear()
    BATCH_EVAL_RUNS.clear()


def test_batch_eval_result_and_distribution_and_failures():
    created = v1_create_dataset(
        DatasetCreateRequest(
            name="eval-ds",
            type="eval_set",
            records=[
                {"input": "a", "output": "b"},
                {"input": "one", "output": "two"},
                {"input": "longer input", "output": "longer output"},
            ],
        )
    )

    run = v1_create_batch_eval(
        BatchEvalCreateRequest(dataset_id=created.dataset_id, criteria=["accuracy", "overall"])
    )

    result = v1_get_batch_eval_result(run.run_id)
    assert result.batch_eval_id == run.run_id
    assert result.summary["total_items"] == 3
    assert "overall" in result.summary["mean_scores"]

    dist = v1_get_batch_eval_distribution(run.run_id, criterion="overall")
    assert dist.criterion == "overall"
    assert sum(dist.buckets.values()) == dist.summary["count"]

    failures = v1_get_batch_eval_failures(run.run_id, limit=5)
    assert failures.batch_eval_id == run.run_id
    assert failures.count >= len(failures.data)
