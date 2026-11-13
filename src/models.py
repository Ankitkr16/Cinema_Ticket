"""
Machine Learning Pipelines and Evaluation Metrics for Cinema Ticket Demand
Provides CinemaDemandPredictor, wrapping ColumnTransformer preprocessor
and regressor with capacity enforcement.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.preprocessing import create_preprocessor, extract_features


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes standard regression and cinema business metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - WAPE: Weighted Absolute Percentage Error (sum(|y - y_hat|) / sum(y) * 100)
    - Capacity violations: count of predictions < 0 or > screen capacity
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    sum_actual = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100) if sum_actual > 0 else 0.0
    
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape_pct": round(wape, 2),
    }
    
    if capacities is not None:
        capacities = np.asarray(capacities, dtype=float)
        violations = int(np.sum((y_pred < 0) | (y_pred > capacities)))
        metrics["capacity_violations"] = violations
        
        # Capacity-adjusted predictions
        clamped_pred = np.clip(y_pred, 0, capacities)
        clamped_mae = float(np.mean(np.abs(y_true - clamped_pred)))
        clamped_rmse = float(np.sqrt(np.mean((y_true - clamped_pred) ** 2)))
        clamped_wape = float((np.sum(np.abs(y_true - clamped_pred)) / sum_actual) * 100) if sum_actual > 0 else 0.0
        
        metrics["clamped_mae"] = round(clamped_mae, 2)
        metrics["clamped_rmse"] = round(clamped_rmse, 2)
        metrics["clamped_wape_pct"] = round(clamped_wape, 2)
        
    return metrics


class CinemaDemandPredictor(BaseEstimator, RegressorMixin):
    """
    Production-ready estimator wrapping preprocessing and regression with
    strict business capacity constraints.
    """
    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 150,
        max_depth: Optional[int] = 10,
        learning_rate: float = 0.08,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.preprocessor_ = create_preprocessor()
        
        if model_type == "random_forest":
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth is not None else 5,
                learning_rate=self.learning_rate,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'random_forest' or 'gradient_boosting'.")
            
        self.feature_names_out_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits preprocessing pipeline and regression model on training data."""
        df_feat = extract_features(X) if "time_slot" not in X.columns else X.copy()
        
        X_trans = self.preprocessor_.fit_transform(df_feat)
        self.model_.fit(X_trans, y)
        
        try:
            self.feature_names_out_ = list(self.preprocessor_.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = []
            
        return self

    def predict(self, X: pd.DataFrame, clamp_to_capacity: bool = True) -> np.ndarray:
        """
        Infers raw demand and optionally clamps between 0 and screen capacity.
        """
        df_feat = extract_features(X) if "time_slot" not in X.columns else X.copy()
        X_trans = self.preprocessor_.transform(df_feat)
        raw_preds = self.model_.predict(X_trans)
        
        if clamp_to_capacity:
            capacities = df_feat["screen_capacity"].values if "screen_capacity" in df_feat.columns else np.full_like(raw_preds, 200.0)
            return np.clip(raw_preds, 0.0, capacities)
            
        return raw_preds

    def predict_detailed(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Produces comprehensive business prediction outputs:
        - raw_prediction
        - predicted_tickets (clamped integer)
        - predicted_occupancy_pct
        - available_seats
        - capacity
        - is_cold_start
        """
        df_feat = extract_features(X) if "time_slot" not in X.columns else X.copy()
        raw_preds = self.predict(df_feat, clamp_to_capacity=False)
        clamped_preds = np.clip(raw_preds, 0.0, df_feat["screen_capacity"].values)
        
        results = pd.DataFrame({
            "raw_prediction": np.round(raw_preds, 2),
            "predicted_tickets": np.round(clamped_preds).astype(int),
            "screen_capacity": df_feat["screen_capacity"].astype(int),
            "predicted_occupancy_percentage": np.round((clamped_preds / df_feat["screen_capacity"]) * 100, 1),
            "available_seats": np.maximum(0, df_feat["screen_capacity"].values - np.round(clamped_preds).astype(int)),
            "is_cold_start": df_feat["is_cold_start"].astype(int)
        })
        return results

