import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
import joblib
raw_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "cinema_shows_raw.csv"))
test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "test.csv"))
model = joblib.load(os.path.join(ROOT_DIR, "models", "best_model.joblib"))
test_df["predicted_tickets"] = np.round(model.predict(test_df, clamp_to_capacity=True))

# 1. Chart 1
daily_trend = raw_df.groupby("show_date")["tickets_sold"].mean().reset_index().sort_values("show_date")
daily_trend["rolling_7"] = daily_trend["tickets_sold"].rolling(7, min_periods=1).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=[str(d) for d in daily_trend["show_date"]],
    y=[round(float(v), 2) for v in daily_trend["tickets_sold"]],
    mode="lines+markers",
    name="Daily Mean Demand"
))
fig1.add_trace(go.Scatter(
    x=[str(d) for d in daily_trend["show_date"]],
    y=[round(float(v), 2) for v in daily_trend["rolling_7"]],
    mode="lines",
    name="7-Day Moving Avg"
))

# 2. Chart 2
mon_thu_df = raw_df[raw_df["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday"])]
fri_sun_df = raw_df[raw_df["weekday"].isin(["Friday", "Saturday", "Sunday"])]
wk_avg = float(mon_thu_df["tickets_sold"].mean())
we_avg = float(fri_sun_df["tickets_sold"].mean())

fig2 = go.Figure(go.Bar(
    x=["Weekday (Mon-Thu)", "Weekend (Fri-Sun)"],
    y=[round(wk_avg, 1), round(we_avg, 1)],
    text=[f"{round(wk_avg, 1)} tickets", f"{round(we_avg, 1)} tickets"],
    textposition="outside",
    marker_color=["#3b82f6", "#6366f1"]
))

# 3. Chart 3
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=["Historical Baseline", "Random Forest", "Gradient Boosting"],
    y=[31.35, 14.75, 12.29],
    name="MAE (Tickets)",
    marker_color="#38bdf8"
))
fig3.add_trace(go.Bar(
    x=["Historical Baseline", "Random Forest", "Gradient Boosting"],
    y=[37.53, 19.51, 15.88],
    name="RMSE (Tickets)",
    marker_color="#6366f1"
))

# 4. Chart 4
test_ordered = test_df.sort_values(by=["show_date", "show_time", "show_id"]).reset_index(drop=True)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=list(range(len(test_ordered))),
    y=[int(v) for v in test_ordered["tickets_sold"]],
    mode="lines+markers",
    name="Actual Tickets"
))
fig4.add_trace(go.Scatter(
    x=list(range(len(test_ordered))),
    y=[round(float(v), 1) for v in test_ordered["predicted_tickets"]],
    mode="lines+markers",
    name="Predicted (Gradient Boosting)"
))

for name, fig in [("Chart 1", fig1), ("Chart 2", fig2), ("Chart 3", fig3), ("Chart 4", fig4)]:
    s = pio.to_json(fig, validate=False)
    has_bdata = "bdata" in s
    parsed = json.loads(s)
    print(f"{name}: has bdata = {has_bdata}")
    for t_idx, t in enumerate(parsed["data"]):
        t_name = t.get("name", "Bar")
        print(f"   Trace {t_idx} ({t_name}): x len={len(t['x'])}, y len={len(t['y'])}, y[:3]={t['y'][:3]}")
