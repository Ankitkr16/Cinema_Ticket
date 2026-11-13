"""
Tests for Data Preprocessing and Feature Generation
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import extract_features, get_time_slot, create_preprocessor, split_chronological


def test_get_time_slot():
    assert get_time_slot("10:30") == "Morning"
    assert get_time_slot("13:45") == "Afternoon"
    assert get_time_slot("17:15") == "Evening"
    assert get_time_slot("20:30") == "Evening"
    assert get_time_slot("22:45") == "Night"


def test_feature_generation():
    raw_data = {
        "show_id": ["SHOW_001", "SHOW_002"],
        "movie_id": ["MOV_001", "MOV_002"],
        "movie_name": ["Movie A", "Movie B"],
        "show_date": ["2026-05-15", "2026-05-18"],
        "show_time": ["17:15", "10:30"],
        "weekday": ["Friday", "Monday"],
        "screen_capacity": [200, 100],
        "ticket_price": [14.0, 10.0],
        "days_since_release": [3, 15],
        "genre": ["Action", "Comedy"],
        "historical_movie_demand": [120.0, np.nan],
        "historical_weekday_demand": [110.0, 80.0],
        "historical_time_slot_demand": [130.0, 70.0],
        "tickets_sold": [145, 60]
    }
    df = pd.DataFrame(raw_data)
    df_feat = extract_features(df)
    
    # Assert generated columns exist
    assert "show_hour" in df_feat.columns
    assert "show_month" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    assert "time_slot" in df_feat.columns
    assert "is_cold_start" in df_feat.columns
    
    # Assert values
    assert df_feat["is_weekend"].iloc[0] == 1  # Friday
    assert df_feat["is_weekend"].iloc[1] == 0  # Monday
    assert df_feat["is_cold_start"].iloc[1] == 1  # NaN historical demand


def test_missing_value_preprocessing_pipeline():
    # Test that preprocessor handles NaN in numerical features via imputer
    data = pd.DataFrame({
        "screen_capacity": [200, 100],
        "ticket_price": [12.0, 15.0],
        "days_since_release": [5, 1],
        "historical_movie_demand": [np.nan, 150.0],  # NaN test
        "historical_weekday_demand": [100.0, np.nan],  # NaN test
        "historical_time_slot_demand": [np.nan, 90.0],  # NaN test
        "show_hour": [17, 10],
        "is_weekend": [1, 0],
        "genre": ["Action", "Drama"],
        "weekday": ["Friday", "Monday"],
        "time_slot": ["Evening", "Morning"]
    })
    
    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(data)
    
    # Transformed output should have no NaNs
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 2


def test_chronological_splitting():
    dates = pd.date_range("2026-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "show_id": [f"SHOW_{i}" for i in range(10)],
        "show_date": dates,
        "show_time": ["10:30"] * 10,
        "tickets_sold": range(10)
    })
    
    train, val, test = split_chronological(df, train_ratio=0.7, val_ratio=0.15)
    
    assert len(train) == 7
    assert len(val) == 1
    assert len(test) == 2
    
    # Ensure strict temporal sequence (max train date < min val date <= min test date)
    assert train["show_date"].max() <= val["show_date"].min()
    assert val["show_date"].max() <= test["show_date"].min()

