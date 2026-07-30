"""ReflexioClient retrieval experiment lifecycle methods."""

from typing import Any

from reflexio import ReflexioClient


def test_retrieval_experiment_client_methods(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, path, kwargs))
        if path.endswith("/results"):
            arm = {
                "arm": "treatment",
                "assigned_user_count": 0,
                "published_session_count": 0,
                "evaluated_user_count": 0,
                "evaluated_session_count": 0,
            }
            return {
                "experiment": {
                    "experiment_id": "exp-1",
                    "holdout_percentage": 10,
                    "started_at": 100,
                },
                "treatment": arm,
                "holdout": {**arm, "arm": "holdout"},
                "unattributed_evaluated_session_count": 0,
            }
        return {"experiments": []}

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    client.list_retrieval_experiments()
    client.start_retrieval_experiment("exp-1", 10)
    client.stop_retrieval_experiment("exp-1")
    results = client.get_retrieval_experiment_results("exp-1")

    assert calls[0][:2] == ("GET", "/api/retrieval_experiments")
    assert calls[1][2]["json"] == {
        "experiment_id": "exp-1",
        "holdout_percentage": 10.0,
    }
    assert calls[2][2]["json"] == {"experiment_id": "exp-1"}
    assert calls[3][:2] == (
        "GET",
        "/api/retrieval_experiments/exp-1/results",
    )
    assert results.experiment.experiment_id == "exp-1"
