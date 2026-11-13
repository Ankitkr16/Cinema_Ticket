"""
Cinema Ticket Demand Forecasting - FastAPI Backend
Provides real inference endpoints for predicting ticket sales,
occupancy rates, and seat availability with cold-start awareness.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator


# Globals for loaded artifacts
model_artifact = None
model_metadata = {}


def load_artifacts():
    global model_artifact, model_metadata
    model_path = os.path.join(ROOT_DIR, "models", "best_model.joblib")
    metadata_path = os.path.join(ROOT_DIR, "models", "model_metadata.json")
    
    if os.path.exists(model_path):
        model_artifact = joblib.load(model_path)
    else:
        print(f"Warning: Model artifact not found at {model_path}.")
        
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            model_metadata = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event to load saved machine learning artifacts on startup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Cinema Ticket Demand Forecasting API",
    description="Real ML inference service predicting cinema ticket sales and occupancy rates.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """Root endpoint for cloud deployment health and discovery."""
    return {
        "service": "Cinema Ticket Demand Forecasting API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }


class ShowPredictionRequest(BaseModel):
    movie_id: str = Field(..., examples=["MOV_001"], description="Unique movie identifier")
    movie_name: str = Field(..., examples=["Galactic Wars: Legacy"], description="Movie title")
    show_date: str = Field(..., examples=["2026-05-15"], description="Show date YYYY-MM-DD")
    show_time: str = Field(..., examples=["20:30"], description="Showtime HH:MM in 24-hr format")
    weekday: Optional[str] = Field(None, examples=["Friday"], description="Day of the week (auto-inferred if omitted)")
    screen_capacity: int = Field(..., gt=0, examples=[220], description="Total seating capacity of the screen")
    ticket_price: float = Field(..., gt=0, examples=[14.50], description="Ticket price in currency units")
    days_since_release: int = Field(..., ge=0, examples=[3], description="Days elapsed since theatrical release")
    genre: str = Field(..., examples=["Sci-Fi"], description="Movie genre")
    historical_movie_demand: Optional[float] = Field(None, examples=[155.0], description="Past average tickets sold for this movie (leave null for cold-start)")
    historical_weekday_demand: Optional[float] = Field(None, examples=[120.0], description="Past average tickets sold on this weekday")
    historical_time_slot_demand: Optional[float] = Field(None, examples=[135.0], description="Past average tickets sold in this time slot")

    @field_validator("show_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("show_date must be in YYYY-MM-DD format.")

    @field_validator("show_time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%H:%M")
            return v
        except ValueError:
            raise ValueError("show_time must be in HH:MM format (e.g. 17:15 or 20:30).")


class ShowPredictionResponse(BaseModel):
    predicted_tickets: int
    predicted_occupancy_percentage: float
    available_seats: int
    capacity: int
    model_used: str
    limitation_or_warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    total_training_records: Optional[int] = None


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint indicating model readiness."""
    global model_artifact, model_metadata
    if model_artifact is None:
        load_artifacts()
        
    is_loaded = model_artifact is not None
    model_name = model_metadata.get("best_model_name", "Unknown")
    model_type = model_metadata.get("model_type", "Unknown")
    train_summary = model_metadata.get("training_data_summary", {})
    records = train_summary.get("train_records", None)
    
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=model_name,
        model_type=model_type,
        total_training_records=records
    )


@app.post("/predict", response_model=ShowPredictionResponse)
def predict(request: ShowPredictionRequest):
    """
    Predicts ticket demand and seat occupancy for an individual cinema show.
    Enforces screen capacity boundaries and flags cold-start uncertainties.
    """
    global model_artifact, model_metadata
    
    if model_artifact is None:
        load_artifacts()
        if model_artifact is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact is not loaded. Train the model first."
            )
            
    # Auto-infer weekday if missing
    parsed_date = datetime.strptime(request.show_date, "%Y-%m-%d")
    weekday_str = request.weekday if request.weekday else parsed_date.strftime("%A")
    
    # Check for cold start conditions
    is_cold_start = (request.historical_movie_demand is None or np.isnan(request.historical_movie_demand) or request.days_since_release <= 2)
    warning_message = None
    if is_cold_start:
        warning_message = (
            "Cold-Start Warning: Movie has limited or zero prior historical show records. "
            "Demand prediction relies on genre and showtime baseline priors with higher statistical uncertainty. "
            "Human managerial review is recommended for screen allocation."
        )
        
    # Construct input DataFrame for inference pipeline
    row_data = {
        "show_id": ["INFER_001"],
        "movie_id": [request.movie_id],
        "movie_name": [request.movie_name],
        "show_date": [request.show_date],
        "show_time": [request.show_time],
        "weekday": [weekday_str],
        "screen_capacity": [request.screen_capacity],
        "ticket_price": [request.ticket_price],
        "days_since_release": [request.days_since_release],
        "genre": [request.genre],
        "historical_movie_demand": [request.historical_movie_demand if request.historical_movie_demand is not None else np.nan],
        "historical_weekday_demand": [request.historical_weekday_demand if request.historical_weekday_demand is not None else np.nan],
        "historical_time_slot_demand": [request.historical_time_slot_demand if request.historical_time_slot_demand is not None else np.nan]
    }
    input_df = pd.DataFrame(row_data)
    
    # Real model execution
    detailed_pred = model_artifact.predict_detailed(input_df)
    
    pred_tickets = int(detailed_pred["predicted_tickets"].iloc[0])
    pred_occupancy = float(detailed_pred["predicted_occupancy_percentage"].iloc[0])
    avail_seats = int(detailed_pred["available_seats"].iloc[0])
    
    model_name = model_metadata.get("best_model_name", "CinemaDemandPredictor")
    
    return ShowPredictionResponse(
        predicted_tickets=pred_tickets,
        predicted_occupancy_percentage=pred_occupancy,
        available_seats=avail_seats,
        capacity=request.screen_capacity,
        model_used=model_name,
        limitation_or_warning=warning_message
    )


if __name__ == "__main__":
    import uvicorn
    # Support Render dynamic PORT or default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
