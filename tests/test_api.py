import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as webapp  # noqa: E402

client = TestClient(webapp.app)


def test_pages_and_config():
    assert client.get("/healthz").json() == {"ok": True}
    r = client.get("/")
    assert r.status_code == 200 and "DepthForge" in r.text and "/static/app.js?v=" in r.text
    cfg = client.get("/api/config").json()
    assert cfg["samples"] and "hybrid" in cfg["methods"] and any(s["key"] == "cascade" for s in cfg["stages"])
    assert client.get("/api/thumb/" + cfg["samples"][0]).headers["content-type"] == "image/jpeg"


def test_run_sample_returns_everything():
    r = client.post("/api/run", data=dict(sample="synthetic_1", scale=2, condition="clean", method="classical"))
    assert r.status_code == 200
    j = r.json()
    assert j["has_gt"] and j["metrics"]["rmse"] > 0 and j["images"]["result"].startswith("data:image/jpeg")
    assert j["view3d"]["w"] > 1 and j["download"].startswith("data:image/png") and j["decisions"]


@pytest.mark.parametrize("form,msg", [
    (dict(sample="nope"), "Unknown sample"),
    (dict(sample="synthetic_1", scale=3), "factor"),
    (dict(sample="synthetic_1", method="magic"), "Unknown method"),
    (dict(), "Choose a sample"),
])
def test_bad_requests_give_clear_errors(form, msg):
    r = client.post("/api/run", data=form)
    assert r.status_code == 400 and msg in r.json()["error"]
