"""
Cinema Ticket Demand Preprocessing and Feature Pipeline
Handles data validation, feature extraction, missing values,
and chronological train/validation/test splitting.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_time_slot(time_str: str) -> str:
    """Classifies show_time string HH:MM into Morning, Afternoon, Evening, Night."""
    hour = int(str(time_str).split(":")[0])
    if hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    elif hour < 21:
        return "Evening"
    else:
        return "Night"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts date, time, and calendar features from raw cinema show records.
    """
    df = df.copy()
    
    # Ensure correct datetime parsing
    df["show_datetime"] = pd.to_datetime(df["show_date"] + " " + df["show_time"])
    df["show_hour"] = df["show_datetime"].dt.hour
    df["show_month"] = df["show_datetime"].dt.month
    df["show_day"] = df["show_datetime"].dt.day
    
    # Binary indicator for weekend (Friday evening / Saturday / Sunday)
    # Weekday column is already present, but is_weekend gives strong tabular signal
    df["is_weekend"] = df["weekday"].isin(["Friday", "Saturday", "Sunday"]).astype(int)
    
    # Categorical time slot
    df["time_slot"] = df["show_time"].apply(get_time_slot)
    
    # Flag cold-start / missing historical demand
    df["is_cold_start"] = df["historical_movie_demand"].isna().astype(int)
    
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs data quality checks: duplicate check, missing values, boundary checks.
    """
    duplicates_count = int(df.duplicated(subset=["show_id"]).sum())
    missing_counts = df.isna().sum().to_dict()
    
    # Capacity violation check in raw data (tickets_sold should never exceed screen_capacity or be negative)
    capacity_violations = int((df["tickets_sold"] > df["screen_capacity"]).sum())
    negative_sales = int((df["tickets_sold"] < 0).sum())
    
    return {
        "total_rows": len(df),
        "duplicates": duplicates_count,
        "missing_counts": missing_counts,
        "capacity_violations": capacity_violations,
        "negative_sales": negative_sales
    }


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split without random shuffling.
    - Earliest 70% -> Train
    - Next 15% -> Validation
    - Final 15% -> Test
    """
    df_sorted = df.sort_values(by=["show_date", "show_time", "show_id"]).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


# Feature definitions for scikit-learn pipeline
NUMERICAL_FEATURES = [
    "screen_capacity",
    "ticket_price",
    "days_since_release",
    "historical_movie_demand",
    "historical_weekday_demand",
    "historical_time_slot_demand",
    "show_hour",
    "is_weekend"
]

CATEGORICAL_FEATURES = [
    "genre",
    "weekday",
    "time_slot"
]


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs Scikit-learn ColumnTransformer for preprocessing.
    - Imputes missing numerical values (using median, with missing indicator)
    - Scales numerical features with StandardScaler
    - One-hot encodes categorical features
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor


def prepare_and_save_splits(
    raw_csv_path: str = "data/raw/cinema_shows_raw.csv",
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw data, performs feature extraction, splits chronologically, and saves splits."""
    df_raw = pd.read_csv(raw_csv_path)
    df_feat = extract_features(df_raw)
    
    train_df, val_df, test_df = split_chronological(df_feat)
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Dataset successfully split chronologically:")
    print(f"  Train records:      {len(train_df)} ({len(train_df)/len(df_feat)*100:.1f}%) | Dates: {train_df['show_date'].min()} to {train_df['show_date'].max()}")
    print(f"  Validation records: {len(val_df)} ({len(val_df)/len(df_feat)*100:.1f}%) | Dates: {val_df['show_date'].min()} to {val_df['show_date'].max()}")
    print(f"  Test records:       {len(test_df)} ({len(test_df)/len(df_feat)*100:.1f}%) | Dates: {test_df['show_date'].min()} to {test_df['show_date'].max()}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_and_save_splits()

