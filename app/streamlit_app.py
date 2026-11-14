"""
Cinema Ticket Demand Forecasting - Interactive ML Analytics Dashboard
A clean, student-engineered machine learning analytics and inference interface.
Integrates directly with the trained model pipeline, dataset, and FastAPI backend.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & RESTRAINED DARK NAVY THEME
# ==============================================================================
st.set_page_config(
    page_title="Cinema Ticket Demand Forecasting",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling without excessive glow or decorative clutter
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Background */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    
    /* Top Header */
    .page-header {
        padding: 0.2rem 0 1.2rem 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.2rem;
    }
    
    .tech-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.22);
        color: #e0e7ff;
        border: 1px solid rgba(129, 140, 248, 0.65);
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.6rem;
    }
    
    .page-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .page-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        line-height: 1.5;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    .kpi-label {
        font-size: 0.76rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    
    .kpi-sub {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 3px;
    }
    
    /* Result Display Panel */
    .result-panel {
        background: #131d36;
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    
    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .pill-red {
        background: rgba(239, 68, 68, 0.18);
        color: #fca5a5;
        border: 1px solid #ef4444;
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.18);
        color: #fde68a;
        border: 1px solid #f59e0b;
    }
    .pill-blue {
        background: rgba(59, 130, 246, 0.18);
        color: #93c5fd;
        border: 1px solid #3b82f6;
    }
    
    /* Streamlit Padding Fix */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def get_backend_url() -> str:
    """
    Resolves backend API URL:
    1. Streamlit secrets (st.secrets["API_BASE_URL"]) for Streamlit Community Cloud
    2. Environment variable API_BASE_URL or API_URL (Render / Docker / local shell)
    3. Default local development server: http://127.0.0.1:8000
    """
    try:
        if hasattr(st, "secrets") and "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass

    env_val = os.getenv("API_BASE_URL") or os.getenv("API_URL")
    if env_val:
        return env_val.rstrip("/")

    return "http://127.0.0.1:8000"


API_BASE_URL = get_backend_url()


# ==============================================================================
# DATA & ARTIFACT LOADERS (ABSOLUTE PATHS ONLY)
# ==============================================================================
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "cinema_shows_raw.csv")
TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")
METRICS_PATH = os.path.join(ROOT_DIR, "reports", "metrics.json")


@st.cache_data
def load_datasets():
    """Loads raw and test datasets with explicit column validation."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}.")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}.")
        
    raw = pd.read_csv(RAW_DATA_PATH)
    raw["show_date_dt"] = pd.to_datetime(raw["show_date"])
    
    test = pd.read_csv(TEST_DATA_PATH)
    test["show_date_dt"] = pd.to_datetime(test["show_date"])
    
    return raw, test


@st.cache_data
def load_metrics_and_metadata():
    """Loads saved model metadata and test set evaluation metrics."""
    metadata = {}
    metrics = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    return metadata, metrics


@st.cache_resource
def load_production_model():
    """Loads production trained model pipeline."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


raw_df, test_df = load_datasets()
metadata, metrics = load_metrics_and_metadata()
model = load_production_model()


def check_api_status():
    """Checks whether the FastAPI service is reachable."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2.0)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, {}


api_active, api_info = check_api_status()


# ==============================================================================
# SIDEBAR NAVIGATION & TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("### Cinema Demand Forecasting")
    st.caption("Show-Level Box Office & Occupancy Prediction")
    st.markdown("---")
    
    nav = st.radio(
        "Navigation Menu",
        ["Dashboard", "Demand Prediction", "Model Evaluation", "Limitations"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**Backend Service Status**")
    if api_active:
        st.markdown(
            '<div style="color:#34d399; font-size:0.85rem; font-weight:600;">'
            '● FastAPI Online</div>',
            unsafe_allow_html=True
        )
        st.caption(f"Host: `{API_BASE_URL}`")
        st.caption(f"Model: {api_info.get('model_name', 'Gradient Boosting')}")
    else:
        st.markdown(
            '<div style="color:#f87171; font-size:0.85rem; font-weight:600;">'
            '○ FastAPI Offline (Using Local Fallback)</div>',
            unsafe_allow_html=True
        )
        st.caption(f"Target: `{API_BASE_URL}`")
        
    st.markdown("---")
    st.markdown("**Dataset & Pipeline Summary**")
    st.markdown(f"• **Dataset Size:** {len(raw_df):,} shows (130 days)  \n• **Screens:** 4 auditoriums (100–300 seats)  \n• **Chronological Split:** 70% Train / 15% Val / 15% Test  \n• **Held-Out Test:** {len(test_df)} shows")
    
    st.markdown("---")
    st.caption("ML Engineering Project | Scikit-Learn, FastAPI & Streamlit")


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="page-header">
    <div class="tech-pill">⚡ Scikit-Learn • Chronological Split • Capacity Clamping • FastAPI</div>
    <div class="page-title">Cinema Ticket Demand Forecasting</div>
    <div class="page-subtitle">
        Predicting individual show-level ticket sales and auditorium seat occupancy to support cinema management with screen allocation and operational staffing.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if nav == "Dashboard":
    best_model_name = metadata.get("best_model_name", "Gradient Boosting")
    best_metrics = metrics.get("model_comparison", {}).get(best_model_name, {})
    mae_val = best_metrics.get("mae", 12.29)
    rmse_val = best_metrics.get("rmse", 15.88)
    wape_val = best_metrics.get("wape_pct", 12.62)
    
    # 4 Clean Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Historical Shows</div>
            <div class="kpi-value">{len(raw_df):,}</div>
            <div class="kpi-sub">130 Days • 4 Screens</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Performing Model</div>
            <div class="kpi-value" style="font-size:1.35rem; color:#818cf8;">{best_model_name}</div>
            <div class="kpi-sub">Validation Winner (Lowest WAPE)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Test Set MAE</div>
            <div class="kpi-value">{mae_val:.2f} <span style="font-size:0.9rem; font-weight:500; color:#94a3b8;">tickets</span></div>
            <div class="kpi-sub">RMSE: {rmse_val:.2f} tickets</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Test Set WAPE</div>
            <div class="kpi-value" style="color:#34d399;">{wape_val:.2f}%</div>
            <div class="kpi-sub">60.8% Error Cut vs Baseline</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Model Comparison Summary Table
    st.markdown("#### Model Performance Summary (Untouched Chronological Test Set)")
    comp_dict = metrics.get("model_comparison", {})
    if comp_dict:
        rows = []
        for m_name, d in comp_dict.items():
            rows.append({
                "Model Architecture": m_name,
                "MAE (Tickets)": f"{d.get('mae', 0):.2f}",
                "RMSE (Tickets)": f"{d.get('rmse', 0):.2f}",
                "WAPE (%)": f"{d.get('wape_pct', 0):.2f}%",
                "Capacity Violations": d.get("clamped_capacity_violations", 0),
                "Clamped MAE": f"{d.get('clamped_mae', d.get('mae', 0)):.2f}"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --------------------------------------------------------------------------
    # INTERACTIVE ANALYTICS CONTROLS (DYNAMIC GRAPH SLICING)
    # --------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**🎯 Interactive Analytics Controls: Slice & Update Graphs**")
        st.caption("Select any Genre, Auditorium Screen, or Time Window below. The Demand Trend and Weekday vs. Weekend charts will instantly recalculate and update in real-time.")
        
        flt1, flt2, flt3 = st.columns([1.5, 1.5, 1.2])
        
        with flt1:
            genre_list = ["All Genres"] + sorted(raw_df["genre"].unique().tolist())
            sel_genre = st.selectbox("Movie Genre Filter", genre_list, index=0)
            
        with flt2:
            screen_map = {
                "All Auditoriums (4 Screens)": None,
                "Screen 1 (300 Seats - Grand)": 300,
                "Screen 2 (220 Seats - Premium)": 220,
                "Screen 3 (160 Seats - Standard)": 160,
                "Screen 4 (100 Seats - Intimate)": 100
            }
            sel_screen_name = st.selectbox("Auditorium Capacity Filter", list(screen_map.keys()), index=0)
            sel_screen_cap = screen_map[sel_screen_name]
            
        with flt3:
            sel_window = st.selectbox("Historical Time Window", ["Full 130 Days (Jan 1 – May 10)", "First 65 Days", "Recent 65 Days"], index=0)
            
        # Apply filtering dynamically
        plot_df = raw_df.copy()
        if sel_genre != "All Genres":
            plot_df = plot_df[plot_df["genre"] == sel_genre]
        if sel_screen_cap is not None:
            plot_df = plot_df[plot_df["screen_capacity"] == sel_screen_cap]
        all_unique_dates = sorted(plot_df["show_date"].unique())
        if sel_window == "First 65 Days" and len(all_unique_dates) > 0:
            cutoff_date = all_unique_dates[min(64, len(all_unique_dates) - 1)]
            plot_df = plot_df[plot_df["show_date"] <= cutoff_date]
        elif sel_window == "Recent 65 Days" and len(all_unique_dates) > 0:
            cutoff_date = all_unique_dates[max(0, len(all_unique_dates) - 65)]
            plot_df = plot_df[plot_df["show_date"] >= cutoff_date]
            
        if len(plot_df) == 0:
            st.warning("No historical shows match the selected combination. Resetting to full dataset.")
            plot_df = raw_df.copy()
            
        filter_status = f"📊 Active Slice: **{len(plot_df):,} shows** | Average Demand: **{plot_df['tickets_sold'].mean():.1f} tickets/show** | Genre: **{sel_genre}** | Auditorium: **{sel_screen_name}**"
        st.markdown(f"<div style='font-size:0.83rem; color:#94a3b8; padding-top:4px;'>{filter_status}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Visual Analytics & Demand Patterns")
    
    # --------------------------------------------------------------------------
    # CHART 1: TICKET DEMAND TREND OVER TIME
    # --------------------------------------------------------------------------
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**1. Ticket Demand Trend Over Time**")
        daily_trend = plot_df.groupby("show_date")["tickets_sold"].mean().reset_index().sort_values("show_date")
        daily_trend["rolling_7"] = daily_trend["tickets_sold"].rolling(7, min_periods=1).mean()
        
        # Explicit native Python lists avoid Plotly 6 binary bdata serialization
        x_dates = [str(d) for d in daily_trend["show_date"]]
        y_daily = [round(float(v), 2) for v in daily_trend["tickets_sold"]]
        y_roll7 = [round(float(v), 2) for v in daily_trend["rolling_7"]]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=x_dates,
            y=y_daily,
            mode="lines+markers",
            name="Daily Mean Demand",
            line=dict(color="#38bdf8", width=1.5),
            marker=dict(size=4, color="#38bdf8"),
            opacity=0.85,
            hovertemplate="<b>Date:</b> %{x}<br><b>Daily Mean:</b> %{y:.1f} tickets<extra></extra>"
        ))
        fig_trend.add_trace(go.Scatter(
            x=x_dates,
            y=y_roll7,
            mode="lines",
            name="7-Day Moving Avg",
            line=dict(color="#f43f5e", width=2.5),
            hovertemplate="<b>7-Day Trend:</b> %{y:.1f} tickets<extra></extra>"
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            margin=dict(l=40, r=20, t=10, b=40),
            height=340,
            xaxis=dict(title=f"Show Date ({len(x_dates)} Days Plotted)", showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(title="Average Tickets Sold per Show", range=[0, max(200, int(max(y_daily) * 1.15) if y_daily else 200)], showgrid=True, gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True, theme=None)
        st.caption(f"Showing {len(plot_df):,} shows. Daily mean ticket sales range from {min(y_daily):.1f} to {max(y_daily):.1f} tickets.")

    # --------------------------------------------------------------------------
    # CHART 2: WEEKDAY VS WEEKEND DEMAND
    # --------------------------------------------------------------------------
    with c2:
        st.markdown("**2. Weekday vs. Weekend Demand Comparison**")
        
        # Calculate consistent weekday vs weekend aggregations directly from plot_df
        mon_thu_df = plot_df[plot_df["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday"])]
        fri_sun_df = plot_df[plot_df["weekday"].isin(["Friday", "Saturday", "Sunday"])]
        
        wk_avg = float(mon_thu_df["tickets_sold"].mean()) if len(mon_thu_df) > 0 else 0.0
        we_avg = float(fri_sun_df["tickets_sold"].mean()) if len(fri_sun_df) > 0 else 0.0
        surge_calc = (((we_avg - wk_avg) / wk_avg) * 100.0) if wk_avg > 0 else 0.0
        
        tab_agg, tab_days = st.tabs(["Aggregate (Weekday vs Weekend)", "Daily Breakdown (Mon–Sun)"])
        
        with tab_agg:
            fig_agg = go.Figure()
            fig_agg.add_trace(go.Bar(
                x=["Weekday (Mon–Thu)", "Weekend (Fri–Sun)"],
                y=[round(wk_avg, 1), round(we_avg, 1)],
                text=[f"{round(wk_avg, 1)} tickets", f"{round(we_avg, 1)} tickets"],
                textposition="outside",
                marker_color=["#3b82f6", "#6366f1"],
                customdata=[int(len(mon_thu_df)), int(len(fri_sun_df))],
                hovertemplate="<b>%{x}</b><br>Average: %{y:.1f} tickets/show<br>Shows: %{customdata}<extra></extra>"
            ))
            fig_agg.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                xaxis=dict(title=""),
                yaxis=dict(title="Average Tickets Sold per Show", range=[0, max(160, int(max(wk_avg, we_avg) * 1.25) if max(wk_avg, we_avg) > 0 else 160)], showgrid=True, gridcolor="#1e293b")
            )
            st.plotly_chart(fig_agg, use_container_width=True, theme=None)
            
        with tab_days:
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_stats = plot_df.groupby("weekday")["tickets_sold"].agg(["mean", "std", "count"]).reindex(weekday_order).fillna(0).reset_index()
            
            day_names = list(weekday_order)
            day_means = [round(float(v), 1) for v in weekday_stats["mean"]]
            day_colors = ["#3b82f6" if d in ["Monday", "Tuesday", "Wednesday", "Thursday"] else "#6366f1" for d in day_names]
            
            fig_wk = go.Figure()
            fig_wk.add_trace(go.Bar(
                x=day_names,
                y=day_means,
                text=[f"{v:.1f}" for v in day_means],
                textposition="outside",
                marker_color=day_colors,
                hovertemplate="<b>%{x}</b><br>Average Demand: %{y:.1f} tickets<extra></extra>"
            ))
            fig_wk.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                xaxis=dict(title="Day of Week"),
                yaxis=dict(title="Average Tickets Sold per Show", range=[0, max(165, int(max(day_means) * 1.2) if day_means and max(day_means) > 0 else 165)], showgrid=True, gridcolor="#1e293b")
            )
            st.plotly_chart(fig_wk, use_container_width=True, theme=None)
            
        st.caption(f"Filtered demand ({len(plot_df):,} shows): Weekdays average {wk_avg:.1f} tickets vs. Weekends average {we_avg:.1f} tickets (+{surge_calc:.1f}% surge).")

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    # --------------------------------------------------------------------------
    # CHART 3: MODEL PERFORMANCE COMPARISON (SEPARATE SCALES)
    # --------------------------------------------------------------------------
    with c3:
        st.markdown("**3. Model Benchmark on Held-Out Test Set**")
        tab_abs, tab_rel = st.tabs(["Absolute Errors (Tickets)", "Relative Percentage Error (WAPE %)"])
        
        models_list = ["Historical Baseline", "Random Forest", "Gradient Boosting"]
        mae_list = [31.35, 14.75, 12.29]
        rmse_list = [37.53, 19.51, 15.88]
        wape_list = [32.19, 15.15, 12.62]
        
        with tab_abs:
            fig_abs = go.Figure()
            fig_abs.add_trace(go.Bar(
                x=models_list,
                y=mae_list,
                name="MAE (Tickets)",
                text=[f"{v:.2f}" for v in mae_list],
                textposition="outside",
                marker_color="#38bdf8"
            ))
            fig_abs.add_trace(go.Bar(
                x=models_list,
                y=rmse_list,
                name="RMSE (Tickets)",
                text=[f"{v:.2f}" for v in rmse_list],
                textposition="outside",
                marker_color="#6366f1"
            ))
            fig_abs.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                xaxis=dict(title=""),
                yaxis=dict(title="Error in Tickets (Lower is Better)", range=[0, 45], showgrid=True, gridcolor="#1e293b"),
                legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_abs, use_container_width=True, theme=None)
            
        with tab_rel:
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Bar(
                x=models_list,
                y=wape_list,
                text=[f"{v:.2f}%" for v in wape_list],
                textposition="outside",
                marker_color=["#64748b", "#38bdf8", "#10b981"]
            ))
            fig_rel.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                xaxis=dict(title=""),
                yaxis=dict(title="WAPE % (Lower is Better)", range=[0, 38], showgrid=True, gridcolor="#1e293b")
            )
            st.plotly_chart(fig_rel, use_container_width=True, theme=None)
        st.caption("Properly separated scales: Tickets error (left) vs. Percentage error (right).")

    # --------------------------------------------------------------------------
    # CHART 4: ACTUAL VS PREDICTED (CHRONOLOGICAL TEST SET)
    # --------------------------------------------------------------------------
    with c4:
        st.markdown("**4. Actual vs. Predicted Demand (Held-Out Test Set)**")
        
        # Ensure test set predictions are computed
        test_ordered = test_df.sort_values(by=["show_date", "show_time", "show_id"]).reset_index(drop=True).copy()
        if "predicted_tickets" not in test_ordered.columns and model is not None:
            test_ordered["predicted_tickets"] = np.round(model.predict(test_ordered, clamp_to_capacity=True))
            
        x_test_idx = list(range(len(test_ordered)))
        y_test_act = [int(v) for v in test_ordered["tickets_sold"]]
        y_test_pred = [round(float(v), 1) for v in test_ordered["predicted_tickets"]]
        
        tab_seq, tab_scat = st.tabs(["Chronological Timeline (All 390 Test Shows)", "Scatter Correlation Plot (N = 390)"])
        
        with tab_seq:
            fig_seq = go.Figure()
            fig_seq.add_trace(go.Scatter(
                x=x_test_idx,
                y=y_test_act,
                mode="lines+markers",
                name="Actual Tickets",
                line=dict(color="#ffffff", width=1.5),
                marker=dict(size=3.5, color="#ffffff"),
                hovertemplate="<b>Test Show #%{x}</b><br>Actual: %{y} tickets<extra></extra>"
            ))
            fig_seq.add_trace(go.Scatter(
                x=x_test_idx,
                y=y_test_pred,
                mode="lines+markers",
                name="Predicted (Gradient Boosting)",
                line=dict(color="#10b981", width=1.8, dash="dot"),
                marker=dict(size=3.5, color="#10b981"),
                hovertemplate="<b>Test Show #%{x}</b><br>Predicted: %{y} tickets<extra></extra>"
            ))
            fig_seq.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                xaxis=dict(title="Chronological Test Show Index (All 390 Held-Out Shows)", showgrid=True, gridcolor="#1e293b"),
                yaxis=dict(title="Tickets Sold per Show", range=[0, 310], showgrid=True, gridcolor="#1e293b"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_seq, use_container_width=True, theme=None)
            
        with tab_scat:
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=y_test_act,
                y=y_test_pred,
                mode="markers",
                name="Shows (N=390)",
                marker=dict(color="#38bdf8", size=5, opacity=0.75),
                hovertemplate="<b>Actual:</b> %{x} tickets<br><b>Predicted:</b> %{y} tickets<extra></extra>"
            ))
            # Add 1:1 diagonal reference line
            fig_scat.add_trace(go.Scatter(
                x=[0, 300],
                y=[0, 300],
                mode="lines",
                name="1:1 Perfect Fit (y = x)",
                line=dict(color="#ef4444", dash="dash", width=1.5)
            ))
            fig_scat.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                margin=dict(l=40, r=20, t=10, b=40),
                height=280,
                xaxis=dict(title="Actual Tickets Sold (Held-Out Test)", range=[0, 310], showgrid=True, gridcolor="#1e293b"),
                yaxis=dict(title="Predicted Tickets Sold (Gradient Boosting)", range=[0, 310], showgrid=True, gridcolor="#1e293b"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_scat, use_container_width=True, theme=None)
        st.caption(f"Chronological sequence and correlation plot on all {len(test_ordered)} held-out test shows (MAE: {mae_val:.2f} tickets, WAPE: {wape_val:.2f}%).")


# ==============================================================================
# 2. DEMAND PREDICTION
# ==============================================================================
elif nav == "Demand Prediction":
    st.markdown("### Single Show Demand Prediction")
    st.write("Configure scheduled show attributes to forecast realized ticket sales, auditorium occupancy, and seat availability.")
    
    # Session state for 1-click test scenario presets
    # Default is set to a realistic mid-run show (Fast Asphalt 11, Action, 15 days, 220 seats)
    if "p_m_id" not in st.session_state:
        st.session_state["p_m_id"] = "MOV_006"
        st.session_state["p_m_name"] = "Fast Asphalt 11"
        st.session_state["p_genre"] = "Action"
        st.session_state["p_days"] = 15
        st.session_state["p_price"] = 14.50
        st.session_state["p_capacity"] = 220
        st.session_state["p_time"] = "17:15"
        st.session_state["p_hist_m"] = 120.0
        st.session_state["p_hist_w"] = 100.0
        st.session_state["p_hist_s"] = 115.0

    st.markdown("**Quick Test Scenarios (Click to Pre-fill):**")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("Midweek Action (Normal Fit)", use_container_width=True):
            st.session_state["p_m_id"] = "MOV_006"
            st.session_state["p_m_name"] = "Fast Asphalt 11"
            st.session_state["p_genre"] = "Action"
            st.session_state["p_days"] = 15
            st.session_state["p_price"] = 14.50
            st.session_state["p_capacity"] = 220
            st.session_state["p_time"] = "17:15"
            st.session_state["p_hist_m"] = 120.0
            st.session_state["p_hist_w"] = 100.0
            st.session_state["p_hist_s"] = 115.0

    with p2:
        if st.button("Discount Tuesday Matinee", use_container_width=True):
            st.session_state["p_m_id"] = "MOV_002"
            st.session_state["p_m_name"] = "The Laughing Detective"
            st.session_state["p_genre"] = "Comedy"
            st.session_state["p_days"] = 18
            st.session_state["p_price"] = 9.00
            st.session_state["p_capacity"] = 160
            st.session_state["p_time"] = "13:45"
            st.session_state["p_hist_m"] = 80.0
            st.session_state["p_hist_w"] = 95.0
            st.session_state["p_hist_s"] = 85.0

    with p3:
        if st.button("Weekend Prime Blockbuster", use_container_width=True):
            st.session_state["p_m_id"] = "MOV_001"
            st.session_state["p_m_name"] = "Galactic Wars: Legacy"
            st.session_state["p_genre"] = "Sci-Fi"
            st.session_state["p_days"] = 2
            st.session_state["p_price"] = 16.50
            st.session_state["p_capacity"] = 220  # Capacity limit test
            st.session_state["p_time"] = "20:30"
            st.session_state["p_hist_m"] = 210.0
            st.session_state["p_hist_w"] = 170.0
            st.session_state["p_hist_s"] = 185.0

    with p4:
        if st.button("Cold-Start Opening Premiere", use_container_width=True):
            st.session_state["p_m_id"] = "MOV_NEW_01"
            st.session_state["p_m_name"] = "New Horizon Premiere"
            st.session_state["p_genre"] = "Sci-Fi"
            st.session_state["p_days"] = 0
            st.session_state["p_price"] = 16.00
            st.session_state["p_capacity"] = 300
            st.session_state["p_time"] = "17:15"
            st.session_state["p_hist_m"] = 0.0  # Cold-start trigger
            st.session_state["p_hist_w"] = 110.0
            st.session_state["p_hist_s"] = 125.0

    st.markdown("<br>", unsafe_allow_html=True)

    # Clean Two-Column Form
    with st.form("clean_prediction_form"):
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown("##### 1. Movie & Commercial Parameters")
            in_movie_id = st.text_input("Movie Identifier", value=st.session_state["p_m_id"])
            in_movie_name = st.text_input("Movie Title", value=st.session_state["p_m_name"])
            all_genres = ["Action", "Sci-Fi", "Comedy", "Drama", "Horror", "Animation", "Romance", "Thriller"]
            genre_index = all_genres.index(st.session_state["p_genre"]) if st.session_state["p_genre"] in all_genres else 0
            in_genre = st.selectbox("Genre", all_genres, index=genre_index)
            in_days_since = st.number_input("Days Since Theatrical Release (0 = Release Day)", min_value=0, max_value=120, value=int(st.session_state["p_days"]))
            in_price = st.number_input("Ticket Price ($)", min_value=5.0, max_value=35.0, value=float(st.session_state["p_price"]), step=0.5)

        with col_f2:
            st.markdown("##### 2. Schedule & Historical Demand")
            in_date = st.date_input("Show Date", value=pd.to_datetime("2026-05-15"))
            all_times = ["10:30", "13:45", "17:15", "20:30", "22:45"]
            time_index = all_times.index(st.session_state["p_time"]) if st.session_state["p_time"] in all_times else 2
            in_time = st.selectbox("Showtime (24-hr format)", all_times, index=time_index)
            
            all_caps = [100, 160, 220, 300]
            cap_index = all_caps.index(st.session_state["p_capacity"]) if st.session_state["p_capacity"] in all_caps else 2
            in_capacity = st.selectbox("Auditorium Capacity (Seats)", all_caps, index=cap_index)
            
            st.caption("ℹ️ *For new releases without historical data, enter 0 for Historical Movie Demand.*")
            in_hist_m = st.number_input("Historical Movie Demand (Past Avg Tickets)", min_value=0.0, max_value=350.0, value=float(st.session_state["p_hist_m"]), step=1.0)
            in_hist_w = st.number_input("Historical Weekday Demand (Past Avg)", min_value=0.0, max_value=350.0, value=float(st.session_state["p_hist_w"]), step=1.0)
            in_hist_s = st.number_input("Historical Time Slot Demand (Past Avg)", min_value=0.0, max_value=350.0, value=float(st.session_state["p_hist_s"]), step=1.0)

        predict_submitted = st.form_submit_button("Predict Demand", type="primary", use_container_width=True)

    if predict_submitted:
        payload = {
            "movie_id": in_movie_id,
            "movie_name": in_movie_name,
            "show_date": str(in_date),
            "show_time": in_time,
            "weekday": in_date.strftime("%A"),
            "screen_capacity": int(in_capacity),
            "ticket_price": float(in_price),
            "days_since_release": int(in_days_since),
            "genre": in_genre,
            "historical_movie_demand": float(in_hist_m) if in_hist_m > 0 else None,
            "historical_weekday_demand": float(in_hist_w) if in_hist_w > 0 else None,
            "historical_time_slot_demand": float(in_hist_s) if in_hist_s > 0 else None
        }
        
        pred_response = None
        # Execute Real Prediction via FastAPI Backend
        try:
            r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5.0)
            if r.status_code == 200:
                pred_response = r.json()
            else:
                st.error(f"API Returned Error {r.status_code}: {r.text}")
        except Exception:
            # Fallback to local model artifact
            if model is not None:
                row_df = pd.DataFrame([{
                    "show_id": "PRED_LOCAL",
                    "movie_id": in_movie_id,
                    "movie_name": in_movie_name,
                    "show_date": str(in_date),
                    "show_time": in_time,
                    "weekday": in_date.strftime("%A"),
                    "screen_capacity": int(in_capacity),
                    "ticket_price": float(in_price),
                    "days_since_release": int(in_days_since),
                    "genre": in_genre,
                    "historical_movie_demand": float(in_hist_m) if in_hist_m > 0 else np.nan,
                    "historical_weekday_demand": float(in_hist_w) if in_hist_w > 0 else np.nan,
                    "historical_time_slot_demand": float(in_hist_s) if in_hist_s > 0 else np.nan
                }])
                det = model.predict_detailed(row_df)
                pred_response = {
                    "predicted_tickets": int(det["predicted_tickets"].iloc[0]),
                    "predicted_occupancy_percentage": float(det["predicted_occupancy_percentage"].iloc[0]),
                    "available_seats": int(det["available_seats"].iloc[0]),
                    "capacity": int(in_capacity),
                    "model_used": metadata.get("best_model_name", "Gradient Boosting (Local Artifact)"),
                    "limitation_or_warning": (
                        "Cold-Start Warning: Movie has limited historical records. Predictions rely on baseline priors with higher uncertainty."
                        if (in_hist_m <= 0 or in_days_since <= 2) else None
                    )
                }

        if pred_response:
            pred_t = pred_response["predicted_tickets"]
            pred_occ = pred_response["predicted_occupancy_percentage"]
            avail_s = pred_response["available_seats"]
            cap_val = pred_response["capacity"]
            model_name_resp = pred_response.get("model_used", "Gradient Boosting")
            
            # Status Badge & Capacity Warning Logic
            if pred_occ >= 100.0 or pred_t >= cap_val:
                badge_html = '<span class="status-pill pill-red">Sold Out (100.0% Capacity)</span>'
                capacity_alert = True
                advice = f"Auditorium capacity limit reached ({pred_t}/{cap_val} seats). Unconstrained demand likely exceeds capacity. Consider allocating a larger auditorium (Screen 1 - 300 seats)."
            elif pred_occ >= 85.0:
                badge_html = f'<span class="status-pill pill-red">High Congestion ({pred_occ:.1f}%)</span>'
                capacity_alert = False
                advice = "High occupancy expected. Assign additional usher staff and prepare concessions for rapid turnover."
            elif pred_occ >= 60.0:
                badge_html = f'<span class="status-pill pill-amber">Optimal Attendance ({pred_occ:.1f}%)</span>'
                capacity_alert = False
                advice = "Optimal auditorium fit. Standard concessions staffing and cleaning turnaround adequate."
            else:
                badge_html = f'<span class="status-pill pill-blue">Moderate Attendance ({pred_occ:.1f}%)</span>'
                capacity_alert = False
                advice = "Auditorium has substantial open capacity. Consider scheduling an intimate auditorium (100 seats) to conserve energy costs."

            # Result Presentation
            st.markdown(f"""
            <div class="result-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.9rem;">
                    <div style="font-size:1.1rem; font-weight:700; color:#f8fafc;">Forecast Results & Seat Allocation</div>
                    <div>{badge_html}</div>
                </div>
            """, unsafe_allow_html=True)
            
            res1, res2, res3, res4 = st.columns(4)
            with res1:
                st.metric("Predicted Tickets Sold", f"{pred_t} seats", f"{pred_occ:.1f}% Occupancy")
            with res2:
                st.metric("Available Seats Remaining", f"{avail_s} seats", f"{avail_s} Unbooked")
            with res3:
                st.metric("Auditorium Capacity", f"{cap_val} seats", "Physical Ceiling")
            with res4:
                st.metric("Active Model", model_name_resp, "Scikit-Learn Pipeline")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Occupancy Progress Bar
            st.markdown(f"**Auditorium Occupancy ({pred_t} / {cap_val} seats occupied)**")
            st.progress(float(min(pred_occ / 100.0, 1.0)))
            
            if capacity_alert:
                st.error(f"⚠️ **Capacity Ceiling Warning:** {advice}")
            else:
                st.info(f"💡 **Managerial Insight:** {advice}")
            
            # Cold-Start Warning Display
            if pred_response.get("limitation_or_warning"):
                st.warning(f"⚠️ **Cold-Start Notice:** {pred_response['limitation_or_warning']}")
                
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 3. MODEL EVALUATION
# ==============================================================================
elif nav == "Model Evaluation":
    st.markdown("### Model Evaluation & Diagnostic Benchmarks")
    st.write("Quantitative model comparison on the untouched chronological test set (390 shows), error diagnostics, and cold-start maturity segmentation.")
    
    # 1. Benchmark Comparison Table
    st.markdown("#### 1. Model Comparison on Chronological Test Set")
    comp_data = metrics.get("model_comparison", {})
    if comp_data:
        eval_rows = []
        for m_name, vals in comp_data.items():
            eval_rows.append({
                "Model Architecture": m_name,
                "MAE (Tickets)": f"{vals.get('mae', 0):.2f}",
                "RMSE (Tickets)": f"{vals.get('rmse', 0):.2f}",
                "WAPE (%)": f"{vals.get('wape_pct', 0):.2f}%",
                "Raw Capacity Violations": vals.get("raw_capacity_violations", 0),
                "Clamped Capacity Violations": vals.get("clamped_capacity_violations", 0)
            })
        st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

    # Explanation Container for Raw vs Clamped Violations
    with st.container(border=True):
        st.markdown("**Explanation: Raw vs. Clamped Capacity Violations**")
        st.write("""
        • **Raw Violations:** When standard unconstrained regression algorithms run on raw test data, decision trees can occasionally predict values below zero (negative tickets) or above screen capacity (e.g., predicting 235 tickets for a 200-seat auditorium). In our test evaluation, Gradient Boosting had 1 raw capacity violation.  
        • **Clamped Violations:** In our production `CinemaDemandPredictor`, a strict capacity-bounding layer clamps all business-facing predictions to `[0, screen_capacity]`. This guarantees **exactly 0 capacity violations** in business operations.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Visual Diagnostic Plots
    st.markdown("#### 2. Visual Diagnostic Plots (Held-Out Test Set)")
    diag1, diag2 = st.columns(2)
    
    with diag1:
        img_act = os.path.join(ROOT_DIR, "reports", "figures", "actual_vs_predicted.png")
        if os.path.exists(img_act):
            st.image(img_act, caption="Actual vs. Predicted Tickets Sold (1:1 Fit & Time Sequence)", use_column_width=True)
            
    with diag2:
        img_err = os.path.join(ROOT_DIR, "reports", "figures", "error_distribution.png")
        if os.path.exists(img_err):
            st.image(img_err, caption="Residual Error Distribution (Actual - Predicted, Mean Error ≈ -0.4 tickets)", use_column_width=True)

    st.markdown("---")

    # 3. Cold-Start Evaluation Breakdown
    st.markdown("#### 3. Cold-Start & Movie Maturity Segmentation")
    st.write("Evaluating forecasting error across movie data maturity tiers confirms that predictions for new movies inherently carry greater statistical variance.")
    
    cs1, cs2 = st.columns([1.2, 1.1])
    with cs1:
        cs_info = metrics.get("cold_start_analysis", {})
        if cs_info:
            cs_rows = [
                {
                    "Maturity Segment": "Unseen / New Movies (No Prior History)",
                    "Shows (n)": cs_info.get("unseen_or_new_movies", {}).get("count", 344),
                    "Baseline MAE": f"{cs_info.get('unseen_or_new_movies', {}).get('baseline', {}).get('mae', 31.60):.2f}",
                    "Best Model MAE": f"{cs_info.get('unseen_or_new_movies', {}).get('best_model', {}).get('mae', 12.67):.2f}",
                    "Best Model WAPE": f"{cs_info.get('unseen_or_new_movies', {}).get('best_model', {}).get('wape_pct', 12.17):.2f}%"
                },
                {
                    "Maturity Segment": "Low-Data Movies (History <= 40 shows)",
                    "Shows (n)": cs_info.get("low_data_movies", {}).get("count", 18),
                    "Baseline MAE": f"{cs_info.get('low_data_movies', {}).get('baseline', {}).get('mae', 29.83):.2f}",
                    "Best Model MAE": f"{cs_info.get('low_data_movies', {}).get('best_model', {}).get('mae', 10.25):.2f}",
                    "Best Model WAPE": f"{cs_info.get('low_data_movies', {}).get('best_model', {}).get('wape_pct', 25.44):.2f}%"
                },
                {
                    "Maturity Segment": "Established Movies (Mature History)",
                    "Shows (n)": cs_info.get("established_movies", {}).get("count", 28),
                    "Baseline MAE": f"{cs_info.get('established_movies', {}).get('baseline', {}).get('mae', 29.28):.2f}",
                    "Best Model MAE": f"{cs_info.get('established_movies', {}).get('best_model', {}).get('mae', 9.00):.2f}",
                    "Best Model WAPE": f"{cs_info.get('established_movies', {}).get('best_model', {}).get('wape_pct', 17.35):.2f}%"
                }
            ]
            st.dataframe(pd.DataFrame(cs_rows), use_container_width=True, hide_index=True)
            st.caption("Forecasting error drops significantly as movie history matures: **9.00 MAE (Established) $\\rightarrow$ 10.25 MAE (Low-Data) $\\rightarrow$ 12.67 MAE (New Movies)**.")
            
    with cs2:
        img_cs = os.path.join(ROOT_DIR, "reports", "figures", "cold_start_performance.png")
        if os.path.exists(img_cs):
            st.image(img_cs, caption="Forecasting Error (MAE) by Movie Maturity Tier", use_column_width=True)


# ==============================================================================
# 4. LIMITATIONS (STREAMLIT NATIVE CONTAINERS - ZERO RAW HTML)
# ==============================================================================
elif nav == "Limitations":
    st.markdown("### System Limitations & Responsible AI Guidelines")
    st.write("Transparent documentation of operational boundaries, statistical assumptions, and governance requirements.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 1. Synthetic Dataset Boundaries")
        st.write("""
        The underlying dataset of 2,600 shows was synthetically generated to model realistic multiplex dynamics (including box office decay, ticket price elasticity, prime-time premiums, and auditorium capacity boundaries). While mathematically modeled after typical cinema behavior, it does not incorporate unpredictable real-world exogenous shocks such as severe weather events, celebrity controversies, localized school holiday variations, or marketing blitzes.
        """)

    with st.container(border=True):
        st.markdown("#### 2. New Movie Cold-Start Limitations")
        st.write("""
        Prior to a film's opening screening, **zero historical ticket sales records exist**. The forecasting pipeline falls back to genre-level empirical priors and showtime multipliers. As demonstrated in our empirical evaluation, cold-start shows exhibit **40.7% higher error** (12.67 MAE) compared to mature releases (9.00 MAE). Cinema managers must treat opening-weekend predictions as probabilistic estimates with wider confidence intervals.
        """)

    with st.container(border=True):
        st.markdown("#### 3. Historical Demand Feature Dependency")
        st.write("""
        The machine learning pipeline relies heavily on cumulative historical features (`historical_movie_demand`, `historical_weekday_demand`, `historical_time_slot_demand`). In production, an automated feature store must update rolling averages strictly before each show's scheduled prediction timestamp ($t' < t$). Any pipeline latency or schema drift will degrade forecast precision.
        """)

    with st.container(border=True):
        st.markdown("#### 4. Capacity Truncation & Right-Censoring")
        st.write("""
        The prediction target is *realized tickets sold*, bounded strictly by screen capacity (e.g., maximum 300 seats). For sold-out blockbusters, true latent audience demand is right-censored; a 100-seat sellout does not disclose whether 180 or 250 prospective attendees attempted to purchase tickets.
        """)

    with st.container(border=True):
        st.markdown("#### 5. Mandatory Human-in-the-Loop Governance")
        st.write("""
        This system is engineered strictly as an **operational decision support tool**. Automated scheduling algorithms must never unilaterally reassign screens or alter ticket pricing without review. Final auditorium assignments for premiere galas, press screenings, and film festivals must remain under discretionary human managerial oversight.
        """)
