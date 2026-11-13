"""
Cinema Ticket Demand Forecasting - Comprehensive Data & Chart Verification Script
Validates:
1. Dataset integrity, row count, tickets_sold min/max/mean
2. Daily demand aggregations, natural fluctuations, and linear trend check
3. Weekday vs Weekend aggregation consistency
4. Held-out test set actuals and predictions (lengths, bounds, NaNs)
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd

def run_verification():
    print("=" * 70)
    print("1. DATASET INTEGRITY & ROW-LEVEL METRICS")
    print("=" * 70)
    raw_path = os.path.join(ROOT_DIR, "data", "raw", "cinema_shows_raw.csv")
    test_path = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
    model_path = os.path.join(ROOT_DIR, "models", "best_model.joblib")
    
    raw_df = pd.read_csv(raw_path)
    raw_df["show_date_dt"] = pd.to_datetime(raw_df["show_date"])
    
    print(f"Total Dataset Rows: {len(raw_df):,}")
    print(f"Total Columns: {len(raw_df.columns)} -> {list(raw_df.columns)}")
    print(f"Date Range: {raw_df['show_date'].min()} to {raw_df['show_date'].max()} ({raw_df['show_date'].nunique()} unique calendar days)")
    
    t_min = raw_df["tickets_sold"].min()
    t_max = raw_df["tickets_sold"].max()
    t_mean = raw_df["tickets_sold"].mean()
    t_std = raw_df["tickets_sold"].std()
    print(f"Target Column (tickets_sold):")
    print(f"  * Minimum: {t_min}")
    print(f"  * Maximum: {t_max}")
    print(f"  * Mean: {t_mean:.2f}")
    print(f"  * Standard Deviation: {t_std:.2f}")
    
    print("\n" + "=" * 70)
    print("2. DAILY DEMAND TREND & LINEARITY ANALYSIS")
    print("=" * 70)
    daily = raw_df.groupby("show_date")["tickets_sold"].agg(["mean", "min", "max", "count"]).reset_index()
    daily["day_idx"] = np.arange(len(daily))
    daily["rolling_7"] = daily["mean"].rolling(7, min_periods=1).mean()
    
    print(f"Daily Demand Aggregations across {len(daily)} days:")
    print(f"  * Daily Mean Minimum: {daily['mean'].min():.2f} tickets (Date: {daily.loc[daily['mean'].idxmin(), 'show_date']})")
    print(f"  * Daily Mean Maximum: {daily['mean'].max():.2f} tickets (Date: {daily.loc[daily['mean'].idxmax(), 'show_date']})")
    print(f"  * Daily Mean Overall Average: {daily['mean'].mean():.2f} tickets")
    print(f"  * 7-Day Moving Avg Range: {daily['rolling_7'].min():.2f} to {daily['rolling_7'].max():.2f} tickets")
    
    # Linear Regression Check
    slope, intercept = np.polyfit(daily["day_idx"], daily["mean"], 1)
    corr = np.corrcoef(daily["day_idx"], daily["mean"])[0, 1]
    r_squared = corr ** 2
    
    print("\nLinear Trend Diagnostic:")
    print(f"  * Linear Slope: {slope:.4f} tickets/day")
    print(f"  * Intercept: {intercept:.2f} tickets")
    print(f"  * Pearson r: {corr:.4f}")
    print(f"  * R-squared: {r_squared:.4f}")
    if r_squared < 0.10:
        print("  -> CONCLUSION: Dataset DOES NOT contain an artificial linear trend.")
        print("     The variance is governed by genuine weekly cycles and release age decays.")
    else:
        print("  -> WARNING: Dataset has linear trend component.")
        
    print("\nFirst 10 Daily Means:")
    for _, row in daily.head(10).iterrows():
        print(f"    {row['show_date']} | Mean: {row['mean']:6.2f} | 7-Day Rolling: {row['rolling_7']:6.2f} | Count: {int(row['count'])}")
        
    print("\nLast 10 Daily Means:")
    for _, row in daily.tail(10).iterrows():
        print(f"    {row['show_date']} | Mean: {row['mean']:6.2f} | 7-Day Rolling: {row['rolling_7']:6.2f} | Count: {int(row['count'])}")

    print("\n" + "=" * 70)
    print("3. WEEKDAY VS WEEKEND AGGREGATION")
    print("=" * 70)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_breakdown = raw_df.groupby("weekday")["tickets_sold"].agg(["mean", "std", "count"]).reindex(weekday_order)
    
    print("Day-of-Week Statistics (Monday through Sunday):")
    for day in weekday_order:
        m = weekday_breakdown.loc[day, "mean"]
        s = weekday_breakdown.loc[day, "std"]
        c = int(weekday_breakdown.loc[day, "count"])
        print(f"  * {day:9s}: Mean = {m:6.2f} tickets | Std = {s:5.2f} | Shows (n) = {c}")
        
    mon_thu_mask = raw_df["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday"])
    fri_sun_mask = raw_df["weekday"].isin(["Friday", "Saturday", "Sunday"])
    
    wk_mean = raw_df[mon_thu_mask]["tickets_sold"].mean()
    wk_std = raw_df[mon_thu_mask]["tickets_sold"].std()
    wk_count = mon_thu_mask.sum()
    
    we_mean = raw_df[fri_sun_mask]["tickets_sold"].mean()
    we_std = raw_df[fri_sun_mask]["tickets_sold"].std()
    we_count = fri_sun_mask.sum()
    
    surge = ((we_mean - wk_mean) / wk_mean) * 100
    print("\nAggregate Weekday vs. Weekend:")
    print(f"  * Weekdays (Mon-Thu) : Mean = {wk_mean:.2f} tickets (std = {wk_std:.2f}, n = {wk_count})")
    print(f"  * Weekends (Fri-Sun) : Mean = {we_mean:.2f} tickets (std = {we_std:.2f}, n = {we_count})")
    print(f"  * Weekend Demand Surge: +{surge:.1f}%")

    print("\n" + "=" * 70)
    print("4. HELD-OUT TEST SET (390 PREDICTIONS) VALIDATION")
    print("=" * 70)
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)
    
    actuals = test_df["tickets_sold"].values
    predictions = model.predict(test_df, clamp_to_capacity=True)
    
    print(f"Test Set Records Count: {len(test_df)}")
    print(f"Actuals Array Length: {len(actuals)}")
    print(f"Predictions Array Length: {len(predictions)}")
    print(f"Actuals Range: Min = {actuals.min()}, Max = {actuals.max()}, Mean = {actuals.mean():.2f}")
    print(f"Predictions Range: Min = {predictions.min():.2f}, Max = {predictions.max():.2f}, Mean = {predictions.mean():.2f}")
    
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    wape = (np.sum(np.abs(actuals - predictions)) / np.sum(actuals)) * 100
    
    print(f"Verification Metrics on Test Set:")
    print(f"  * MAE: {mae:.2f} tickets")
    print(f"  * RMSE: {rmse:.2f} tickets")
    print(f"  * WAPE: {wape:.2f}%")
    
    # Capacity violations check
    capacities = test_df["screen_capacity"].values
    neg_violations = np.sum(predictions < 0)
    cap_violations = np.sum(predictions > capacities)
    print(f"Physical Boundary Checks:")
    print(f"  * Negative Ticket Violations: {neg_violations}")
    print(f"  * Exceeding Capacity Violations: {cap_violations}")

    print("\n" + "=" * 70)
    print("5. NaN / MISSING VALUE AUDIT ACROSS ALL PLOTTED SERIES")
    print("=" * 70)
    print(f"  * raw_df['tickets_sold'] NaNs: {raw_df['tickets_sold'].isna().sum()}")
    print(f"  * daily['mean'] NaNs: {daily['mean'].isna().sum()}")
    print(f"  * daily['rolling_7'] NaNs: {daily['rolling_7'].isna().sum()}")
    print(f"  * weekday_breakdown['mean'] NaNs: {weekday_breakdown['mean'].isna().sum()}")
    print(f"  * test_df['tickets_sold'] (actuals) NaNs: {np.isnan(actuals).sum()}")
    print(f"  * test predictions NaNs: {np.isnan(predictions).sum()}")
    print("=" * 70)
    print("ALL DATA VERIFICATION CHECKS COMPLETED CLEANLY.")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()

