from asgi_lifespan import LifespanManager
import pytest
from httpx import AsyncClient
from httpx import ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_health():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_analyze_single():
    payload = {"text": "Reliance Industries reports record profits", "ticker": "RELIANCE"}
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/analyze", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["label"] in ["bullish", "bearish", "neutral"]
            assert 0 <= data["confidence"] <= 1
            assert data["ticker"] == "RELIANCE"

@pytest.mark.asyncio
async def test_analyze_batch():
    payload = {"texts": ["Good results", "Bad news"], "ticker": "TCS"}
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/analyze/batch", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "results" in data
            assert data["count"] == 2

@pytest.mark.asyncio
async def test_analyze_batch_limit():
    payload = {"texts": ["x"] * 51, "ticker": "TCS"}
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/analyze/batch", json=payload)
            assert resp.status_code == 400
            assert "Max 50 texts" in resp.text

@pytest.mark.asyncio
async def test_validation_error():
    payload = {"ticker": "TCS"}  # missing 'text'
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/analyze", json=payload)
            assert resp.status_code == 422