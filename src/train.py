"""
Training and Model Selection Pipeline
Trains Baseline, Random Forest, and Gradient Boosting models on the chronological training set,
evaluates on the validation set, selects the best model, and saves artifacts with Joblib.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any
from src.preprocessing import prepare_and_save_splits, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.baseline import CapacityAdjustedHistoricalBaseline
from src.models import CinemaDemandPredictor, compute_metrics


def train_and_select_model(
    data_dir: str = "data/processed",
    models_dir: str = "models"
) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("Processed splits not found. Preparing chronological splits...")
        train_df, val_df, _ = prepare_and_save_splits()
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
    X_train = train_df.drop(columns=["tickets_sold"])
    y_train = train_df["tickets_sold"]
    
    X_val = val_df.drop(columns=["tickets_sold"])
    y_val = val_df["tickets_sold"]
    val_capacities = val_df["screen_capacity"].values
    
    print("\n" + "="*60)
    print("STEP 1: Training Capacity-Adjusted Historical Baseline...")
    print("="*60)
    baseline_model = CapacityAdjustedHistoricalBaseline()
    baseline_model.fit(X_train, y_train)
    val_preds_base = baseline_model.predict(X_val)
    metrics_base = compute_metrics(y_val, val_preds_base, val_capacities)
    print(f"Baseline Validation Metrics: MAE={metrics_base['mae']}, RMSE={metrics_base['rmse']}, WAPE={metrics_base['wape_pct']}%")
    
    print("\n" + "="*60)
    print("STEP 2: Training RandomForestRegressor Pipeline...")
    print("="*60)
    rf_predictor = CinemaDemandPredictor(
        model_type="random_forest",
        n_estimators=160,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
    rf_predictor.fit(X_train, y_train)
    val_preds_rf = rf_predictor.predict(X_val, clamp_to_capacity=True)
    metrics_rf = compute_metrics(y_val, val_preds_rf, val_capacities)
    print(f"Random Forest Validation Metrics: MAE={metrics_rf['mae']}, RMSE={metrics_rf['rmse']}, WAPE={metrics_rf['wape_pct']}%")
    
    print("\n" + "="*60)
    print("STEP 3: Training GradientBoostingRegressor Pipeline...")
    print("="*60)
    gb_predictor = CinemaDemandPredictor(
        model_type="gradient_boosting",
        n_estimators=160,
        max_depth=5,
        learning_rate=0.07,
        min_samples_split=5,
        random_state=42
    )
    gb_predictor.fit(X_train, y_train)
    val_preds_gb = gb_predictor.predict(X_val, clamp_to_capacity=True)
    metrics_gb = compute_metrics(y_val, val_preds_gb, val_capacities)
    print(f"Gradient Boosting Validation Metrics: MAE={metrics_gb['mae']}, RMSE={metrics_gb['rmse']}, WAPE={metrics_gb['wape_pct']}%")
    
    # Model Selection strictly based on Validation WAPE & MAE
    val_comparison = {
        "Baseline": metrics_base,
        "Random Forest": metrics_rf,
        "Gradient Boosting": metrics_gb
    }
    
    candidates = [
        ("Random Forest", rf_predictor, metrics_rf),
        ("Gradient Boosting", gb_predictor, metrics_gb)
    ]
    # Pick the model with the lowest validation MAE
    candidates.sort(key=lambda x: x[2]["mae"])
    best_name, best_predictor, best_metrics = candidates[0]
    
    print("\n" + "="*60)
    print(f"MODEL SELECTION RESULT: Best Model is '{best_name}'")
    print(f"Validation MAE: {best_metrics['mae']}, RMSE: {best_metrics['rmse']}, WAPE: {best_metrics['wape_pct']}%")
    print("="*60)
    
    # Save best model artifact and baseline
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    baseline_path = os.path.join(models_dir, "baseline_model.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    
    joblib.dump(best_predictor, best_model_path)
    joblib.dump(baseline_model, baseline_path)
    
    metadata = {
        "best_model_name": best_name,
        "model_type": best_predictor.model_type,
        "validation_metrics": best_metrics,
        "validation_comparison": val_comparison,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "target": "tickets_sold"
        },
        "training_data_summary": {
            "train_records": len(train_df),
            "train_date_range": [str(train_df["show_date"].min()), str(train_df["show_date"].max())],
            "validation_records": len(val_df),
            "validation_date_range": [str(val_df["show_date"].min()), str(val_df["show_date"].max())]
        },
        "model_hyperparameters": {
            "n_estimators": best_predictor.n_estimators,
            "max_depth": best_predictor.max_depth,
            "min_samples_split": best_predictor.min_samples_split,
            "random_state": best_predictor.random_state
        }
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSaved best model to: {best_model_path}")
    print(f"Saved baseline model to: {baseline_path}")
    print(f"Saved metadata to: {metadata_path}")
    
    return metadata


if __name__ == "__main__":
    train_and_select_model()

