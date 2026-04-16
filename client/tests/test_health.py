"""Smoke test: is the gate reachable?"""
import pytest


@pytest.mark.smoke
def test_health_returns_ok(client):
    resp = client.health()
    assert resp["status"] == "ok"
