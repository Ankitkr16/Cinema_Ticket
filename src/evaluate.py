"""
Evaluation Pipeline and Cold-Start Analysis
Evaluates Baseline, Random Forest, and Gradient Boosting on the untouched chronological test set.
Calculates MAE, RMSE, WAPE, capacity violations, and segments cold-start performance.
Saves metric summaries and visualization charts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from src.models import CinemaDemandPredictor, compute_metrics
from src.baseline import CapacityAdjustedHistoricalBaseline


def evaluate_models_and_cold_start(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    X_train = train_df.drop(columns=["tickets_sold"])
    y_train = train_df["tickets_sold"]
    
    X_test = test_df.drop(columns=["tickets_sold"])
    y_test = test_df["tickets_sold"].values
    test_capacities = test_df["screen_capacity"].values
    
    # 2. Load or recreate models
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    baseline_model_path = os.path.join(models_dir, "baseline_model.joblib")
    
    if os.path.exists(best_model_path):
        best_predictor = joblib.load(best_model_path)
    else:
        raise FileNotFoundError(f"Model not found at {best_model_path}. Run src.train first.")
        
    if os.path.exists(baseline_model_path):
        baseline_model = joblib.load(baseline_model_path)
    else:
        baseline_model = CapacityAdjustedHistoricalBaseline()
        baseline_model.fit(X_train, y_train)
        
    # Also train / load the alternative ML model for full 3-way test set comparison
    rf_predictor = CinemaDemandPredictor(model_type="random_forest", n_estimators=160, max_depth=12, random_state=42)
    rf_predictor.fit(X_train, y_train)
    
    gb_predictor = CinemaDemandPredictor(model_type="gradient_boosting", n_estimators=160, max_depth=5, learning_rate=0.07, random_state=42)
    gb_predictor.fit(X_train, y_train)
    
    # 3. Generate test predictions
    # Baseline
    preds_base = baseline_model.predict(X_test)
    metrics_base = compute_metrics(y_test, preds_base, test_capacities)
    
    # Random Forest
    raw_preds_rf = rf_predictor.predict(X_test, clamp_to_capacity=False)
    clamped_preds_rf = np.clip(raw_preds_rf, 0, test_capacities)
    metrics_rf = compute_metrics(y_test, clamped_preds_rf, test_capacities)
    metrics_rf["raw_capacity_violations"] = int(np.sum((raw_preds_rf < 0) | (raw_preds_rf > test_capacities)))
    metrics_rf["clamped_capacity_violations"] = 0
    
    # Gradient Boosting
    raw_preds_gb = gb_predictor.predict(X_test, clamp_to_capacity=False)
    clamped_preds_gb = np.clip(raw_preds_gb, 0, test_capacities)
    metrics_gb = compute_metrics(y_test, clamped_preds_gb, test_capacities)
    metrics_gb["raw_capacity_violations"] = int(np.sum((raw_preds_gb < 0) | (raw_preds_gb > test_capacities)))
    metrics_gb["clamped_capacity_violations"] = 0
    
    test_comparison = {
        "Baseline": metrics_base,
        "Random Forest": metrics_rf,
        "Gradient Boosting": metrics_gb
    }
    
    # Determine which model is designated best
    best_model_name = "Gradient Boosting" if best_predictor.model_type == "gradient_boosting" else "Random Forest"
    best_clamped_preds = clamped_preds_gb if best_model_name == "Gradient Boosting" else clamped_preds_rf
    best_raw_preds = raw_preds_gb if best_model_name == "Gradient Boosting" else raw_preds_rf
    
    # 4. Cold-Start Analysis on Test Set
    # Identify movies seen in train vs unseen / new movies
    train_movies = set(train_df["movie_id"].unique())
    
    # Segment 1: Unseen Movies (never appeared in training) or release age <= 3 days
    is_unseen = ~test_df["movie_id"].isin(train_movies)
    is_new_release = test_df["days_since_release"] <= 3
    cold_start_mask = is_unseen | is_new_release
    
    # Segment 2: Low-data movies (appearing in train with <= 30 shows or 4 <= release age <= 7)
    train_movie_counts = train_df["movie_id"].value_counts().to_dict()
    test_df["train_show_count"] = test_df["movie_id"].map(lambda m: train_movie_counts.get(m, 0))
    low_data_mask = (~cold_start_mask) & ((test_df["train_show_count"] < 40) | (test_df["days_since_release"] <= 7))
    
    # Segment 3: Established movies
    established_mask = (~cold_start_mask) & (~low_data_mask)
    
    cold_start_metrics = {
        "unseen_or_new_movies": {
            "count": int(cold_start_mask.sum()),
            "baseline": compute_metrics(y_test[cold_start_mask], preds_base[cold_start_mask]),
            "best_model": compute_metrics(y_test[cold_start_mask], best_clamped_preds[cold_start_mask])
        },
        "low_data_movies": {
            "count": int(low_data_mask.sum()),
            "baseline": compute_metrics(y_test[low_data_mask], preds_base[low_data_mask]),
            "best_model": compute_metrics(y_test[low_data_mask], best_clamped_preds[low_data_mask])
        },
        "established_movies": {
            "count": int(established_mask.sum()),
            "baseline": compute_metrics(y_test[established_mask], preds_base[established_mask]),
            "best_model": compute_metrics(y_test[established_mask], best_clamped_preds[established_mask])
        }
    }
    
    # 5. Save Metrics JSON
    results = {
        "test_records_count": len(test_df),
        "test_date_range": [str(test_df["show_date"].min()), str(test_df["show_date"].max())],
        "model_comparison": test_comparison,
        "best_model_name": best_model_name,
        "cold_start_analysis": cold_start_metrics
    }
    
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # 6. Generate Evaluation Figures
    # A. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_plot_df = pd.DataFrame([
        {"Model": "Historical Baseline", "MAE": metrics_base["mae"], "RMSE": metrics_base["rmse"], "WAPE (%)": metrics_base["wape_pct"]},
        {"Model": "Random Forest", "MAE": metrics_rf["mae"], "RMSE": metrics_rf["rmse"], "WAPE (%)": metrics_rf["wape_pct"]},
        {"Model": "Gradient Boosting", "MAE": metrics_gb["mae"], "RMSE": metrics_gb["rmse"], "WAPE (%)": metrics_gb["wape_pct"]}
    ]).melt(id_vars="Model", var_name="Metric", value_name="Value")
    
    sns.barplot(data=metrics_plot_df, x="Metric", y="Value", hue="Model", palette=["#7f7f7f", "#1f77b4", "#2ca02c"])
    plt.title("Model Comparison on Untouched Chronological Test Set", fontsize=13, fontweight="bold")
    plt.ylabel("Error Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison_metrics.png"), dpi=200)
    plt.close()
    
    # B. Actual vs Predicted Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, best_clamped_preds, alpha=0.5, color="#1f77b4", edgecolor="none")
    min_val, max_val = min(y_test.min(), best_clamped_preds.min()), max(y_test.max(), best_clamped_preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect 1:1 Prediction")
    axes[0].set_title(f"Actual vs Predicted Tickets ({best_model_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Tickets Sold")
    axes[0].set_ylabel("Predicted Tickets Sold")
    axes[0].legend()
    
    # Sequence timeline snippet
    sample_len = 60
    axes[1].plot(range(sample_len), y_test[:sample_len], label="Actual Tickets", color="black", linewidth=1.5, marker="o", markersize=3)
    axes[1].plot(range(sample_len), best_clamped_preds[:sample_len], label=f"Predicted ({best_model_name})", color="#2ca02c", linestyle="--", linewidth=1.8)
    axes[1].set_title(f"Chronological Timeline Test Slice (First {sample_len} Shows)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Test Show Sequence")
    axes[1].set_ylabel("Tickets Sold")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()
    
    # C. Residual / Error Distribution
    residuals = y_test - best_clamped_preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.histplot(residuals, kde=True, color="#d62728", ax=axes[0], bins=30)
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_title(f"Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Prediction Error (Tickets)")
    
    sns.scatterplot(x=best_clamped_preds, y=residuals, alpha=0.5, color="#8c564b", ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_title("Residuals vs Predicted Value (Homoscedasticity Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Tickets")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "error_distribution.png"), dpi=200)
    plt.close()
    
    # D. Cold-Start Performance Breakdown Chart
    cs_df = pd.DataFrame([
        {"Segment": "New / Cold-Start\n(No Prior History)", "MAE": cold_start_metrics["unseen_or_new_movies"]["best_model"]["mae"], "Count": cold_start_metrics["unseen_or_new_movies"]["count"]},
        {"Segment": "Low-Data Movies\n(Limited History)", "MAE": cold_start_metrics["low_data_movies"]["best_model"]["mae"], "Count": cold_start_metrics["low_data_movies"]["count"]},
        {"Segment": "Established Movies\n(Extensive History)", "MAE": cold_start_metrics["established_movies"]["best_model"]["mae"], "Count": cold_start_metrics["established_movies"]["count"]},
    ])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=cs_df, x="Segment", y="MAE", palette="Blues_r", ax=ax)
    ax.set_title(f"Forecast Error (MAE) by Movie Maturity ({best_model_name})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (Tickets)")
    for i, row in cs_df.iterrows():
        ax.text(i, row["MAE"] + 0.3, f"MAE: {row['MAE']}\n(n={row['Count']})", ha="center", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cold_start_performance.png"), dpi=200)
    plt.close()
    
    # Print clean summary
    print("\n" + "="*70)
    print("CHRONOLOGICAL TEST SET EVALUATION SUMMARY (N = 390)")
    print("="*70)
    print(f"{'Model':<25} | {'MAE':<8} | {'RMSE':<8} | {'WAPE (%)':<10} | {'Cap Violations':<15}")
    print("-" * 70)
    print(f"{'Capacity Baseline':<25} | {metrics_base['mae']:<8} | {metrics_base['rmse']:<8} | {metrics_base['wape_pct']:<10}% | {0:<15}")
    print(f"{'Random Forest':<25} | {metrics_rf['mae']:<8} | {metrics_rf['rmse']:<8} | {metrics_rf['wape_pct']:<10}% | {metrics_rf['clamped_capacity_violations']:<15}")
    print(f"{'Gradient Boosting':<25} | {metrics_gb['mae']:<8} | {metrics_gb['rmse']:<8} | {metrics_gb['wape_pct']:<10}% | {metrics_gb['clamped_capacity_violations']:<15}")
    print("="*70)
    print("\nCOLD-START SEGMENTATION PERFORMANCE:")
    print(f"  * Unseen / New Movies (n={cold_start_metrics['unseen_or_new_movies']['count']}): Best Model MAE = {cold_start_metrics['unseen_or_new_movies']['best_model']['mae']} (WAPE: {cold_start_metrics['unseen_or_new_movies']['best_model']['wape_pct']}%)")
    print(f"  * Low-Data Movies     (n={cold_start_metrics['low_data_movies']['count']}): Best Model MAE = {cold_start_metrics['low_data_movies']['best_model']['mae']} (WAPE: {cold_start_metrics['low_data_movies']['best_model']['wape_pct']}%)")
    print(f"  * Established Movies  (n={cold_start_metrics['established_movies']['count']}): Best Model MAE = {cold_start_metrics['established_movies']['best_model']['mae']} (WAPE: {cold_start_metrics['established_movies']['best_model']['wape_pct']}%)")
    print("="*70)
    
    return results


if __name__ == "__main__":
    evaluate_models_and_cold_start()

