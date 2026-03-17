"""Integration test: upload -> correlation -> preprocess. Uses Flask test client."""
import os
import pytest


@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def examples_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "examples")


def test_get_index(client):
    r = client.get("/")
    assert r.status_code == 200


def test_upload_correlation_preprocess(client, examples_dir):
    csv_path = os.path.join(examples_dir, "iris.csv")
    if not os.path.exists(csv_path):
        pytest.skip("examples/iris.csv not found")
    with open(csv_path, "rb") as f:
        r = client.post("/upload", data={"file": (f, "iris.csv")})
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    data = r.get_json()
    assert "numcols" in data

    r = client.post(
        "/correlationMatrices",
        json={"colsIgnore": "all", "dropMissing": "none", "imputeStrategy": "none", "dropZero": "none"},
        content_type="application/json",
    )
    assert r.status_code == 200

    r = client.post(
        "/preprocess",
        json={"indicators": [0, 1, 2, 3], "predictors": [4], "stratify": 0},
        content_type="application/json",
    )
    assert r.status_code == 200
    out = r.get_json()
    assert "predictors" in out and "indicators" in out
