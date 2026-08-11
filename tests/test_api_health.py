from fastapi.testclient import TestClient

from main import app


def test_health_endpoint_shape():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "qdrant" in data
    assert "neo4j" in data
    assert "ollama" in data
    assert data["status"] in {
        "ok",
        "degraded",
        "unavailable",
        "READY",
        "DEGRADED",
        "FAILED",
        "NOT_CONFIGURED",
    }
    assert "components" in data
