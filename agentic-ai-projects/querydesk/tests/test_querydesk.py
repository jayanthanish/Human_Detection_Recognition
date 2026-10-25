import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.storage import Store, DataError
from app.agent import run_agent
from app.models import ModelError


def tool(name, **args):
    return {
        "tool_calls": [{"function": {"name": name, "arguments": args}}],
        "content": "",
    }


def final(**data):
    return {"content": json.dumps(data)}


class ScriptedModel:
    model = "test"

    def __init__(self, steps):
        self.steps = iter(steps)
        self.messages = []

    def tool_chat(self, messages, tools):
        self.messages = list(messages)
        return next(self.steps)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def dataset(store):
    return store.import_csv(
        "sales.csv", b"region,revenue,cost\nWest,100,40\nEast,80,30\nWest,50,20\n"
    )


def test_csv_profile_and_normalized_headers(store):
    meta = store.import_csv(
        "x.csv", b"Region Name,Amount,Amount,ZIP\nWest,12.5,3,00123\n"
    )
    assert [c["name"] for c in meta["columns"]] == [
        "region_name",
        "amount",
        "amount_2",
        "zip",
    ]
    assert meta["columns"][-1]["type"] == "TEXT"
    assert store.execute(meta["id"], "SELECT zip FROM data")["rows"] == [["00123"]]


@pytest.mark.parametrize("content", [b"", b"a,b\n1\n", b"a\n", b"\xff"])
def test_invalid_csv(store, content):
    with pytest.raises(DataError):
        store.import_csv("bad.csv", content)


def test_real_sql_aggregation(store, dataset):
    result = store.execute(
        dataset["id"],
        "SELECT region, SUM(revenue-cost) AS profit FROM data GROUP BY region ORDER BY profit DESC",
    )
    assert result["rows"] == [["West", 90.0], ["East", 50.0]]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE data",
        "UPDATE data SET revenue=0",
        "SELECT * FROM data; DELETE FROM data",
        "PRAGMA database_list",
        'ATTACH DATABASE "/tmp/x" AS x',
        "SELECT * FROM metadata",
        "SELECT * FROM sqlite_master",
        'SELECT load_extension("x")',
    ],
)
def test_query_controls_reject_non_data_access(store, dataset, sql):
    with pytest.raises(DataError):
        store.execute(dataset["id"], sql)


def test_row_limit(store, dataset):
    assert (
        store.execute(dataset["id"], "SELECT * FROM data", row_limit=1)["truncated"]
        is True
    )


def test_cte_works(store, dataset):
    assert store.execute(
        dataset["id"],
        "WITH x AS (SELECT SUM(revenue) AS total FROM data) SELECT total FROM x",
    )["rows"] == [[230.0]]


def test_agent_recovers_from_sql_error(store, dataset):
    ai = ScriptedModel(
        [
            tool("inspect_dataset"),
            tool("run_sql", sql="SELECT missing FROM data"),
            tool("run_sql", sql="SELECT SUM(revenue) AS total FROM data"),
            final(explanation="Total revenue.", needs_context=False),
        ]
    )
    result = run_agent(store, dataset["id"], "Total revenue?", ai)
    assert result["rows"] == [[230.0]]
    assert [t["status"] for t in result["trace"]] == ["ok", "error", "ok"]
    assert any("missing" in m["content"] for m in ai.messages if m["role"] == "tool")


def test_agent_requires_schema_before_sql(store, dataset):
    ai = ScriptedModel(
        [
            tool("run_sql", sql="SELECT 1"),
            final(explanation="Cannot answer.", needs_context=True),
        ]
    )
    result = run_agent(store, dataset["id"], "Question?", ai)
    assert result["trace"][0]["status"] == "error"


def test_agent_budget(store, dataset):
    ai = ScriptedModel([tool("inspect_dataset")] * 5)
    with pytest.raises(ModelError):
        run_agent(store, dataset["id"], "Question?", ai, max_calls=2)


def test_api_upload_ask_history_and_export(tmp_path):
    ai = ScriptedModel(
        [
            tool("inspect_dataset"),
            tool(
                "run_sql",
                sql="SELECT region, SUM(revenue) AS total FROM data GROUP BY region",
            ),
            final(explanation="Revenue by region.", needs_context=False),
        ]
    )
    client = TestClient(create_app(tmp_path, ai))
    r = client.post(
        "/api/datasets",
        files={
            "file": ("sales.csv", b"region,revenue\nWest,100\nEast,50\n", "text/csv")
        },
    )
    assert r.status_code == 200
    did = r.json()["id"]
    assert client.get("/api/datasets/" + did).status_code == 200
    r = client.post(
        "/api/datasets/" + did + "/ask", json={"question": "Revenue by region?"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == [["East", 50.0], ["West", 100.0]]
    history = client.get("/api/datasets/" + did + "/history").json()
    assert len(history) == 1
    export = client.get(
        "/api/datasets/" + did + "/history/" + r.json()["id"] + "/export"
    )
    assert "region,total" in export.text
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
