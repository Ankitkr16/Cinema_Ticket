"""
Tests for FastAPI Backend Endpoints
"""

import pytest
import httpx

# Compatibility patch for Starlette 0.36 + httpx 0.28+
_orig_init = httpx.Client.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("app", None)
    return _orig_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_init

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "model_name" in data
    assert data["model_loaded"] is True


def test_api_prediction_endpoint_valid(client):
    payload = {
        "movie_id": "MOV_001",
        "movie_name": "Galactic Wars: Legacy",
        "show_date": "2026-05-15",
        "show_time": "20:30",
        "weekday": "Friday",
        "screen_capacity": 220,
        "ticket_price": 14.50,
        "days_since_release": 7,
        "genre": "Sci-Fi",
        "historical_movie_demand": 160.0,
        "historical_weekday_demand": 130.0,
        "historical_time_slot_demand": 145.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assert output schema and business constraints
    assert "predicted_tickets" in data
    assert "predicted_occupancy_percentage" in data
    assert "available_seats" in data
    assert "capacity" in data
    assert data["capacity"] == 220
    assert 0 <= data["predicted_tickets"] <= 220
    assert 0.0 <= data["predicted_occupancy_percentage"] <= 100.0
    assert data["available_seats"] == 220 - data["predicted_tickets"]


def test_api_prediction_endpoint_cold_start(client):
    # Brand new movie with None historical demand and 0 days since release
    payload = {
        "movie_id": "MOV_BRAND_NEW",
        "movie_name": "New Blockbuster Premiere",
        "show_date": "2026-06-01",
        "show_time": "17:15",
        "weekday": "Monday",
        "screen_capacity": 300,
        "ticket_price": 16.00,
        "days_since_release": 0,
        "genre": "Action",
        "historical_movie_demand": None,
        "historical_weekday_demand": 100.0,
        "historical_time_slot_demand": 120.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert 0 <= data["predicted_tickets"] <= 300
    assert data["limitation_or_warning"] is not None
    assert "Cold-Start Warning" in data["limitation_or_warning"]


def test_api_invalid_date_validation(client):
    payload = {
        "movie_id": "MOV_001",
        "movie_name": "Test Movie",
        "show_date": "15-05-2026",  # Invalid format (should be YYYY-MM-DD)
        "show_time": "20:30",
        "screen_capacity": 150,
        "ticket_price": 12.0,
        "days_since_release": 5,
        "genre": "Action"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_invalid_negative_capacity(client):
    payload = {
        "movie_id": "MOV_001",
        "movie_name": "Test Movie",
        "show_date": "2026-05-15",
        "show_time": "20:30",
        "screen_capacity": -50,  # Negative capacity invalid
        "ticket_price": 12.0,
        "days_since_release": 5,
        "genre": "Action"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
