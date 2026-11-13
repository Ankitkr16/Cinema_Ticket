"""
Tests for Machine Learning Models, Baseline, and Capacity Boundaries
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from src.models import CinemaDemandPredictor, compute_metrics
from src.baseline import CapacityAdjustedHistoricalBaseline


def test_saved_model_loading():
    model_path = os.path.join("models", "best_model.joblib")
    assert os.path.exists(model_path), "Saved model artifact models/best_model.joblib must exist."
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model must implement predict method."
    assert hasattr(model, "predict_detailed"), "Loaded model must implement predict_detailed method."


def test_prediction_output_validation():
    model_path = os.path.join("models", "best_model.joblib")
    model = joblib.load(model_path)
    
    sample_df = pd.DataFrame([{
        "show_id": "TEST_001",
        "movie_id": "MOV_001",
        "movie_name": "Galactic Wars: Legacy",
        "show_date": "2026-05-15",
        "show_time": "20:30",
        "weekday": "Friday",
        "screen_capacity": 160,
        "ticket_price": 14.5,
        "days_since_release": 5,
        "genre": "Sci-Fi",
        "historical_movie_demand": 140.0,
        "historical_weekday_demand": 120.0,
        "historical_time_slot_demand": 135.0
    }])
    
    preds = model.predict(sample_df)
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float, np.floating))
    
    detailed = model.predict_detailed(sample_df)
    assert "predicted_tickets" in detailed.columns
    assert "predicted_occupancy_percentage" in detailed.columns
    assert "available_seats" in detailed.columns
    assert detailed["predicted_tickets"].iloc[0] <= 160


def test_non_negative_predictions():
    model_path = os.path.join("models", "best_model.joblib")
    model = joblib.load(model_path)
    
    # Extreme scenario: very old flop with expensive ticket price
    sample_df = pd.DataFrame([{
        "show_id": "TEST_002",
        "movie_id": "MOV_999",
        "movie_name": "Unpopular Movie",
        "show_date": "2026-05-15",
        "show_time": "10:30",
        "weekday": "Monday",
        "screen_capacity": 100,
        "ticket_price": 40.0,
        "days_since_release": 90,
        "genre": "Drama",
        "historical_movie_demand": 0.0,
        "historical_weekday_demand": 20.0,
        "historical_time_slot_demand": 15.0
    }])
    
    preds = model.predict(sample_df, clamp_to_capacity=True)
    assert preds[0] >= 0.0, "Predictions must never be negative."


def test_predictions_not_exceeding_screen_capacity():
    model_path = os.path.join("models", "best_model.joblib")
    model = joblib.load(model_path)
    
    # Extreme scenario: massive blockbuster scheduled in a tiny 50-seat screen
    tiny_capacity = 50
    sample_df = pd.DataFrame([{
        "show_id": "TEST_003",
        "movie_id": "MOV_001",
        "movie_name": "Galactic Wars: Legacy",
        "show_date": "2026-05-15",
        "show_time": "20:30",
        "weekday": "Saturday",
        "screen_capacity": tiny_capacity,
        "ticket_price": 10.0,
        "days_since_release": 1,
        "genre": "Sci-Fi",
        "historical_movie_demand": 250.0,
        "historical_weekday_demand": 190.0,
        "historical_time_slot_demand": 210.0
    }])
    
    preds = model.predict(sample_df, clamp_to_capacity=True)
    assert preds[0] <= tiny_capacity, f"Prediction {preds[0]} exceeded screen capacity {tiny_capacity}."
    
    detailed = model.predict_detailed(sample_df)
    assert detailed["predicted_tickets"].iloc[0] <= tiny_capacity
    assert detailed["available_seats"].iloc[0] >= 0
    assert detailed["predicted_occupancy_percentage"].iloc[0] <= 100.0


def test_baseline_cold_start_fallback():
    baseline = CapacityAdjustedHistoricalBaseline()
    train_dummy = pd.DataFrame({
        "movie_id": ["MOV_A", "MOV_A", "MOV_B"],
        "genre": ["Action", "Action", "Comedy"],
        "weekday": ["Friday", "Saturday", "Friday"],
        "time_slot": ["Evening", "Evening", "Morning"],
        "screen_capacity": [200, 200, 100],
        "tickets_sold": [150, 170, 50]
    })
    baseline.fit(train_dummy)
    
    # Completely brand new movie MOV_NEW never in train_dummy
    test_new = pd.DataFrame([{
        "movie_id": "MOV_NEW",
        "genre": "Action",
        "weekday": "Friday",
        "time_slot": "Evening",
        "screen_capacity": 150,
        "historical_movie_demand": np.nan
    }])
    pred = baseline.predict(test_new)
    assert len(pred) == 1
    assert 0 <= pred[0] <= 150

