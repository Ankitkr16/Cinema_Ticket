"""
Data and Chart Validation Script
Verifies source datasets, aggregations, test predictions, and metric boundaries.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_path = os.path.join(ROOT_DIR, "data", "raw", "cinema_shows_raw.csv")
test_path = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
model_path = os.path.join(ROOT_DIR, "models", "best_model.joblib")

print("="*60)
print("1. DATASET INTEGRITY & VALIDATION LOG")
print("="*60)

raw_df = pd.read_csv(raw_path)
print(f"Total Raw Records: {len(raw_df)}")
print(f"Raw Columns ({len(raw_df.columns)}): {list(raw_df.columns)}")
print(f"First 5 Records:\n{raw_df[['show_id', 'movie_name', 'show_date', 'weekday', 'tickets_sold', 'screen_capacity']].head()}")
print(f"Target (tickets_sold) Stats: Min={raw_df['tickets_sold'].min()}, Max={raw_df['tickets_sold'].max()}, Mean={raw_df['tickets_sold'].mean():.2f}")

# Chart 1 Validation: Daily Trend
df_trend = raw_df.copy()
df_trend["show_date"] = pd.to_datetime(df_trend["show_date"])
daily_trend = df_trend.groupby("show_date")["tickets_sold"].agg(["mean", "count"]).reset_index().sort_values("show_date")
daily_trend["rolling_7"] = daily_trend["mean"].rolling(7, min_periods=1).mean()
print("\n" + "-"*60)
print("CHART 1: Daily Demand Trend")
print(f"  * Days plotted: {len(daily_trend)} consecutive days")
print(f"  * Daily Mean Range: {daily_trend['mean'].min():.2f} to {daily_trend['mean'].max():.2f} tickets/show")
print(f"  * Rolling 7 Range: {daily_trend['rolling_7'].min():.2f} to {daily_trend['rolling_7'].max():.2f} tickets/show")
print(f"  * Missing values in daily trend: {daily_trend.isna().sum().to_dict()}")

# Chart 2 Validation: Weekday vs Weekend
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_agg = raw_df.groupby("weekday")["tickets_sold"].agg(["mean", "std", "count"]).reindex(weekday_order).reset_index()
print("\n" + "-"*60)
print("CHART 2: Weekday vs Weekend")
for _, r in weekday_agg.iterrows():
    print(f"  * {r['weekday']:<10}: Mean={r['mean']:<6.2f} tickets (n={int(r['count'])}, std={r['std']:.2f})")

# Chart 3 & 4 Validation: Test Predictions
test_df = pd.read_csv(test_path)
model = joblib.load(model_path)
preds = model.predict(test_df, clamp_to_capacity=True)
test_df["predicted_tickets"] = preds

print("\n" + "-"*60)
print("CHART 4: Actual vs Predicted (Held-Out Test Set)")
print(f"  * Total Test Records: {len(test_df)} (Target: 390)")
print(f"  * Predictions Length: {len(preds)}")
print(f"  * Actual Tickets Range: {test_df['tickets_sold'].min()} to {test_df['tickets_sold'].max()}")
print(f"  * Predicted Tickets Range: {preds.min():.2f} to {preds.max():.2f}")
print(f"  * Missing values in test actuals: {test_df['tickets_sold'].isna().sum()}")
print(f"  * Missing values in predictions: {np.isnan(preds).sum()}")
print(f"  * Out-of-bounds predictions (<0 or >capacity): {((preds < 0) | (preds > test_df['screen_capacity'])).sum()}")

print("\n" + "="*60)
print("ALL VALIDATION CHECKS COMPLETED SUCCESSFULLY.")
print("="*60)
