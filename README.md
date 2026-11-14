# Cinema Ticket Demand Forecasting System

A complete, production-grade Machine Learning system for predicting individual show-level cinema ticket demand and seat occupancy. The project enables cinema multiplex managers to optimize auditorium screen allocation, prevent capacity bottlenecks, and balance operational staffing.

---

## Architecture Overview

```
                                [ Scheduled Show Parameters ]
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │   Data Preprocessing    │
                                 │   & Feature Pipeline    │
                                 │ (Zero Leakage Enc / Imp)│
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ GradientBoostingRegressor│
                                 │   (Trained Production)  │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Capacity Clamping Layer │
                                 │  [0, screen_capacity]   │
                                 └────────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          ┌───────────────────────┐                       ┌───────────────────────┐
          │     FastAPI Backend   │ ◄─── REST API ──────  │  Streamlit Dashboard  │
          │  (/health, /predict)  │                       │  (Inference & Metrics)│
          └───────────────────────┘                       └───────────────────────┘
```

---

## Folder Structure

```
Cinema_Ticket/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API endpoints (/health, /predict)
├── app/
│   └── streamlit_app.py            # Streamlit dashboard (Executive, Predict, Eval, Limits)
├── data/
│   ├── raw/
│   │   ├── cinema_shows_raw.csv    # 2,600 synthetic show records
│   │   └── dataset_metadata.json   # Generation metadata and audit trail
│   └── processed/
│       ├── train.csv               # Earliest 70% of chronological records (1,819 shows)
│       ├── val.csv                 # Middle 15% of records (391 shows)
│       └── test.csv                # Final 15% of records (390 shows)
├── models/
│   ├── baseline_model.joblib       # Fitted capacity-adjusted historical baseline
│   ├── best_model.joblib           # Trained production model pipeline artifact
│   └── model_metadata.json         # Training metrics, features, hyperparameters
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis Jupyter Notebook
├── reports/
│   ├── figures/                    # 12 visualization figures (distributions, heatmaps, etc.)
│   ├── evaluation_report.md        # 17-section technical ML evaluation report
│   └── metrics.json                # Exact test set and cold-start metrics
├── scripts/
│   ├── test_json_fix.py            # Serialized JSON wire verification script
│   ├── validate_charts.py          # Dashboard aggregation validation script
│   └── verify_all_data.py          # Full dataset and metrics integrity audit script
├── src/
│   ├── __init__.py
│   ├── baseline.py                 # Capacity-Adjusted Historical Baseline Estimator
│   ├── data_generator.py           # Synthetic dataset generator with zero data leakage
│   ├── eda_generator.py            # Generates EDA plots and Jupyter notebook
│   ├── evaluate.py                 # Evaluates models on held-out test set & cold-start
│   ├── models.py                   # CinemaDemandPredictor pipeline & metrics
│   ├── preprocessing.py            # Feature extraction, validation, ColumnTransformer
│   └── train.py                    # Chronological training and validation model selection
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # FastAPI endpoint tests (health, predict, validation)
│   ├── test_data_and_preprocessing.py # Preprocessing, imputation, and feature tests
│   └── test_models.py              # Model loading, capacity clamping, non-negative tests
├── .gitignore                      # Git exclusion rules with explicit ML artifact tracking
├── .python-version                 # Pinned Python version (3.11.9) for cloud buildpacks
├── .streamlit/
│   ├── config.toml                 # Streamlit cloud server & dark theme configuration
│   └── secrets.toml.example        # Cloud secrets template for API_BASE_URL
├── render.yaml                     # Render Infrastructure-as-Code Web Service blueprint
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Comprehensive setup and deployment guide
```

---

## Installation Instructions

### 1. Clone the repository and enter directory
```bash
git clone <repo-url>
cd Cinema_Ticket
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

### 1. Generate the Synthetic Dataset
Generates 2,600 show records chronologically with strictly zero data leakage:
```bash
python -m src.data_generator
```

### 2. Run Data Preprocessing and Model Training
Splits dataset chronologically (70% Train, 15% Val, 15% Test), trains Baseline, Random Forest, and Gradient Boosting, and saves the best model:
```bash
python -m src.train
```

### 3. Evaluate Models on Held-Out Test Set
Evaluates models against the untouched chronological test set, conducts cold-start analysis, and saves evaluation plots:
```bash
python -m src.evaluate
```

### 4. Start the FastAPI Backend
Launches the REST API server:
```bash
uvicorn api.main:app --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Start the Streamlit Frontend Dashboard
Launches the interactive operations dashboard:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 6. Run the Pytest Test Suite
Runs all 14 unit and integration tests across data processing, models, and API endpoints:
```bash
pytest -v
```

---

## Cloud Deployment Guide (GitHub, Render & Streamlit Cloud)

The architecture is decoupled into a **FastAPI backend** hosted on [Render](https://render.com) and an **interactive Streamlit frontend** hosted on [Streamlit Community Cloud](https://streamlit.io/cloud).

```
┌─────────────────────────────────┐           REST API           ┌─────────────────────────────────┐
│   Streamlit Community Cloud     │ ───────────────────────────► │         Render Backend          │
│   (app/streamlit_app.py)        │   (API_BASE_URL: /predict)   │       (api/main.py:app)         │
│   https://<app>.streamlit.app   │ ◄─────────────────────────── │ https://<api>.onrender.com      │
└─────────────────────────────────┘           JSON Response      └─────────────────────────────────┘
```

### 1. Push Project to GitHub

1. Stage all project files, verify `.gitignore`, and commit:
   ```bash
   git add .
   git commit -m "feat: complete CineCast AI deployment readiness"
   ```
2. Create a new repository on [GitHub](https://github.com/new) (e.g., `cinema-ticket-demand-forecasting`).
3. Set default branch to `main`, link remote, and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/cinema-ticket-demand-forecasting.git
   git push -u origin main
   ```

---

### 2. Deploy FastAPI Backend on Render

1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository (`cinema-ticket-demand-forecasting`).
4. Configure the service settings:
   * **Name:** `cinecast-ai-api`
   * **Region:** Any preferred region (e.g., Oregon, Frankfurt, Singapore)
   * **Branch:** `main`
   * **Root Directory:** Leave empty (root)
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. In **Environment Variables**, add:
   * `PYTHON_VERSION` = `3.11.9`
6. Set **Health Check Path** to `/health`.
7. Click **Create Web Service**.
   * *Alternative (1-Click Blueprint):* Render will automatically detect `render.yaml` in the repo root if you choose **New + > Blueprint**.
8. Once deployed, copy your public service URL:
   * Production API: `https://cinecast-api.onrender.com`
   * Test health check: `curl https://cinecast-api.onrender.com/health`

---

### 3. Deploy Streamlit Frontend on Streamlit Community Cloud

1. Log into [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **New app**.
3. Select your GitHub repository, branch (`main`), and set the main file path:
   * **Repository:** `Ankitkr16/Cinema_Ticket`
   * **Branch:** `main`
   * **Main file path:** `app/streamlit_app.py`
4. Click **Advanced settings** (or go to **App Settings > Secrets** after creating):
5. Add the backend connection URL in the **Secrets** editor:
   ```toml
   API_BASE_URL = "https://cinecast-api.onrender.com"
   ```
6. Click **Deploy!**
7. Your interactive dashboard is now live (e.g., `https://cinecast-ai.streamlit.app`).

---

### 4. Dynamic API_BASE_URL Resolution & Local Fallback

The application intelligently resolves the backend URL in three tiers:
1. **Cloud Production (Streamlit Cloud):** Reads `st.secrets["API_BASE_URL"]`.
2. **Container / Shell Override:** Reads `os.getenv("API_BASE_URL")` or `os.getenv("API_URL")`.
3. **Local Development Fallback:** Defaults automatically to `http://127.0.0.1:8000`.

*If the backend is ever unreachable or waking up from Render cold sleep, the Streamlit frontend gracefully activates its **built-in local predictor fallback** using `models/best_model.joblib`, ensuring zero downtime or broken UI states.*

---

## Example API Request and Response

### Endpoint: `POST /predict`

#### Sample Request (`curl`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

#### Sample Response (`200 OK`)
```json
{
  "predicted_tickets": 178,
  "predicted_occupancy_percentage": 80.9,
  "available_seats": 42,
  "capacity": 220,
  "model_used": "Gradient Boosting",
  "limitation_or_warning": null
}
```

### Cold-Start Sample (Brand New Movie)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "movie_id": "MOV_NEW",
       "movie_name": "Premiere Night Film",
       "show_date": "2026-06-01",
       "show_time": "17:15",
       "weekday": "Friday",
       "screen_capacity": 300,
       "ticket_price": 16.00,
       "days_since_release": 0,
       "genre": "Action",
       "historical_movie_demand": null,
       "historical_weekday_demand": 110.0,
       "historical_time_slot_demand": 135.0
     }'
```

#### Response (`200 OK` with Warning)
```json
{
  "predicted_tickets": 195,
  "predicted_occupancy_percentage": 65.0,
  "available_seats": 105,
  "capacity": 300,
  "model_used": "Gradient Boosting",
  "limitation_or_warning": "Cold-Start Warning: Movie has limited or zero prior historical show records. Demand prediction relies on genre and showtime baseline priors with higher statistical uncertainty. Human managerial review is recommended for screen allocation."
}
```

---

## Empirical Benchmark Summary

Evaluated on the 390 untouched chronological test records:

| Model | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Capacity Violations |
|---|---|---|---|---|
| **Capacity Baseline** | 31.35 | 37.53 | 32.19% | 0 |
| **Random Forest** | 14.75 | 19.51 | 15.15% | 0 |
| **Gradient Boosting (Production)** | **12.29** | **15.88** | **12.62%** | **0** |

### Cold-Start Segmentation
- **Unseen / New Movies (No History):** MAE = 12.67 (WAPE: 12.17%)
- **Low-Data Movies (Limited History):** MAE = 10.25 (WAPE: 25.44%)
- **Established Movies (Extensive History):** MAE = 9.00 (WAPE: 17.35%)

---

## Limitations and Operational Guidance

1. **Synthetic Nature:** Synthetically generated data simulates price elasticity, weekend premiums, time slot multipliers, and decay curves, but cannot replicate exogenous shocks such as severe weather or viral marketing shifts.
2. **Cold-Start Uncertainty:** New movie predictions exhibit higher variance. Managerial oversight is recommended for premiere weekends.
3. **Capacity Truncation:** The target reflects actual tickets sold (bounded by capacity), which understates unconstrained latent demand for sell-out shows.
4. **Human Review Requirement:** The forecasting system should serve as decision-support for multiplex managers rather than an unmonitored automated scheduler.

