"""
Capacity-Adjusted Historical Baseline Model
Estimates expected ticket sales using historical demand for comparable movies,
weekdays, and time slots learned strictly from training data.
Enforces screen capacity boundaries and provides fallback for cold-start movies.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Any, Optional


class CapacityAdjustedHistoricalBaseline(BaseEstimator, RegressorMixin):
    """
    Capacity-Adjusted Historical Baseline Estimator.
    Learns empirical historical priors from training data:
    - Global average demand
    - Movie average demand
    - Genre average demand (fallback for new movies)
    - Weekday adjustment factors
    - Time slot adjustment factors
    
    Predictions are strictly clamped between 0 and screen capacity.
    """
    def __init__(self, fallback_strategy: str = "genre_slot_blend"):
        self.fallback_strategy = fallback_strategy
        self.global_mean_: float = 0.0
        self.movie_means_: Dict[str, float] = {}
        self.genre_means_: Dict[str, float] = {}
        self.weekday_factors_: Dict[str, float] = {}
        self.slot_factors_: Dict[str, float] = {}
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns historical statistics strictly from training records."""
        df = X.copy()
        if y is not None:
            df["tickets_sold"] = y
        elif "tickets_sold" not in df.columns:
            raise ValueError("Target 'tickets_sold' must be provided in X or y.")
            
        self.global_mean_ = float(df["tickets_sold"].mean())
        
        # Movie empirical means
        self.movie_means_ = df.groupby("movie_id")["tickets_sold"].mean().to_dict()
        
        # Genre empirical means (fallback for unseen / cold-start movies)
        self.genre_means_ = df.groupby("genre")["tickets_sold"].mean().to_dict()
        
        # Weekday adjustment factors relative to global mean
        weekday_means = df.groupby("weekday")["tickets_sold"].mean()
        self.weekday_factors_ = {k: float(v / self.global_mean_) for k, v in weekday_means.items()}
        
        # Time slot adjustment factors relative to global mean
        slot_col = "time_slot" if "time_slot" in df.columns else "show_time"
        slot_means = df.groupby(slot_col)["tickets_sold"].mean()
        self.slot_factors_ = {k: float(v / self.global_mean_) for k, v in slot_means.items()}
        
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates capacity-bounded predictions.
        Clamped to [0, screen_capacity] for business validity.
        """
        if not self.fitted_:
            raise RuntimeError("Baseline model is not fitted yet.")
            
        predictions = []
        for _, row in X.iterrows():
            m_id = row.get("movie_id", None)
            genre = row.get("genre", None)
            weekday = row.get("weekday", None)
            slot = row.get("time_slot", None)
            if slot is None and "show_time" in row:
                from src.preprocessing import get_time_slot
                slot = get_time_slot(row["show_time"])
                
            capacity = float(row.get("screen_capacity", 200))
            
            # 1. Determine base demand estimation
            # Prioritize row's historical_movie_demand if available
            hist_movie = row.get("historical_movie_demand", np.nan)
            if pd.notna(hist_movie):
                base_est = float(hist_movie)
            elif m_id in self.movie_means_:
                base_est = self.movie_means_[m_id]
            elif genre in self.genre_means_:
                # Fallback for new movies: genre average
                base_est = self.genre_means_[genre]
            else:
                # Ultimate fallback: global average
                base_est = self.global_mean_
                
            # 2. Apply weekday and showtime adjustment factors
            w_factor = self.weekday_factors_.get(weekday, 1.0)
            s_factor = self.slot_factors_.get(slot, 1.0)
            
            raw_pred = base_est * (w_factor * 0.5 + 0.5) * (s_factor * 0.5 + 0.5)
            
            # 3. Enforce screen capacity bounds [0, screen_capacity]
            bounded_pred = np.clip(raw_pred, 0.0, capacity)
            predictions.append(bounded_pred)
            
        return np.array(predictions)

