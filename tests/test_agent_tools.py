from app import main
from app.schemas import DatasetCreateRequest


def setup_function():
    main.REQUEST_ID.set("rid-agent")
    main.DATASETS.clear()
    main.RUNBOOK_RUNS.clear()


def test_agent_tools_marks_allowed_by_scope():
    main.AUTH_CONTEXT.set({"role": "analyst", "scopes": ["agent:invoke", "datasets:read", "metrics:read"], "api_key": "analyst-key"})
    payload = main.v1_agent_tools()
    assert payload["request_id"] == "rid-agent"
    rows = {row["tool"]: row for row in payload["data"]}
    assert rows["datasets.list"]["allowed"] is True
    assert rows["metrics.dashboard"]["allowed"] is True
    assert rows["runbooks.list"]["allowed"] is False


def test_agent_run_executes_requested_tools():
    created = main.v1_create_dataset(
        DatasetCreateRequest(name="agent-ds", type="eval_set", records=[{"input": "x", "output": "y"}])
    )
    assert created.dataset_id
    main.AUTH_CONTEXT.set({"role": "analyst", "scopes": ["agent:invoke", "datasets:read", "metrics:read"], "api_key": "analyst-key"})

    payload = main.v1_agent_run(
        {
            "goal": "show datasets and metrics",
            "requested_tools": ["datasets.list", "metrics.dashboard"],
            "tool_args": {"datasets.list": {"limit": 5}},
        }
    )
    assert payload["summary"]["total_tools"] == 2
    assert payload["summary"]["failed_tools"] == 0
    tools = {item["tool"]: item for item in payload["execution"]}
    assert tools["datasets.list"]["result"]["ok"] is True
    assert tools["metrics.dashboard"]["result"]["ok"] is True


def test_agent_run_rejects_forbidden_tool():
    main.AUTH_CONTEXT.set({"role": "analyst", "scopes": ["agent:invoke", "datasets:read"], "api_key": "analyst-key"})
    payload = main.v1_agent_run({"goal": "list runbooks", "requested_tools": ["runbooks.list"]})
    assert payload["summary"]["failed_tools"] == 1
    assert payload["execution"][0]["result"]["error"] == "forbidden_tool"
