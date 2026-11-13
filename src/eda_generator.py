"""
EDA Script and Notebook Generator
Generates all exploratory data analysis figures in reports/figures/
and exports a reproducible Jupyter notebook to notebooks/eda.ipynb.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_eda_figures_and_notebook(
    raw_path: str = "data/raw/cinema_shows_raw.csv",
    figures_dir: str = "reports/figures",
    notebook_path: str = "notebooks/eda.ipynb"
):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    df = pd.read_csv(raw_path)
    df["occupancy_pct"] = (df["tickets_sold"] / df["screen_capacity"]) * 100
    df["is_weekend_cat"] = df["weekday"].apply(lambda w: "Weekend (Fri-Sun)" if w in ["Friday", "Saturday", "Sunday"] else "Weekday (Mon-Thu)")
    
    # 1. Ticket Sales & Occupancy Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["tickets_sold"], kde=True, color="#1f77b4", ax=axes[0], bins=30)
    axes[0].set_title("Ticket Sales Distribution per Show", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Tickets Sold")
    axes[0].set_ylabel("Show Count")
    
    sns.histplot(df["occupancy_pct"], kde=True, color="#2ca02c", ax=axes[1], bins=30)
    axes[1].set_title("Seat Occupancy Rate Distribution (%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Occupancy Percentage (%)")
    axes[1].set_ylabel("Show Count")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "ticket_sales_distribution.png")
    fig.savefig(fig1_path, dpi=200)
    plt.close()
    
    # 2. Weekday vs Weekend Demand
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=df, x="weekday", y="tickets_sold", order=weekday_order, palette="Blues_d", ax=axes[0])
    axes[0].set_title("Demand by Day of Week", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Day of Week")
    axes[0].set_ylabel("Tickets Sold")
    axes[0].tick_params(axis='x', rotation=30)
    
    sns.barplot(data=df, x="is_weekend_cat", y="tickets_sold", palette=["#4c72b0", "#dd8452"], ax=axes[1], errorbar="sd")
    axes[1].set_title("Weekday vs Weekend Mean Demand (with Std Dev)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Mean Tickets Sold")
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "weekday_vs_weekend_demand.png")
    fig.savefig(fig2_path, dpi=200)
    plt.close()
    
    # 3. Movie-wise Demand
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    movie_demand = df.groupby(["movie_name", "genre"])["tickets_sold"].mean().reset_index().sort_values(by="tickets_sold", ascending=False)
    sns.barplot(data=movie_demand, x="tickets_sold", y="movie_name", hue="genre", dodge=False, ax=axes[0], palette="tab10")
    axes[0].set_title("Mean Tickets Sold per Movie (Colored by Genre)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mean Tickets Sold")
    axes[0].set_ylabel("Movie Title")
    axes[0].legend(title="Genre", loc="lower right")
    
    genre_demand = df.groupby("genre")["tickets_sold"].mean().reset_index().sort_values(by="tickets_sold", ascending=False)
    sns.barplot(data=genre_demand, x="genre", y="tickets_sold", palette="magma", ax=axes[1])
    axes[1].set_title("Mean Ticket Demand by Genre", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Genre")
    axes[1].set_ylabel("Mean Tickets Sold")
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "moviewise_demand.png")
    fig.savefig(fig3_path, dpi=200)
    plt.close()
    
    # 4. Showtime-wise Demand
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    time_order = ["10:30", "13:45", "17:15", "20:30", "22:45"]
    sns.boxplot(data=df, x="show_time", y="tickets_sold", order=time_order, palette="Spectral", ax=axes[0])
    axes[0].set_title("Demand Distribution Across Showtimes", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Showtime (24-hour)")
    axes[0].set_ylabel("Tickets Sold")
    
    time_genre = df.pivot_table(index="genre", columns="show_time", values="tickets_sold", aggfunc="mean")[time_order]
    sns.heatmap(time_genre, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Mean Demand Heatmap: Genre vs Showtime", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Showtime")
    axes[1].set_ylabel("Genre")
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "showtime_demand.png")
    fig.savefig(fig4_path, dpi=200)
    plt.close()
    
    # 5. Screen Capacity vs Ticket Sales
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x="screen_capacity", y="tickets_sold", hue="genre", alpha=0.6, ax=axes[0], palette="tab10")
    # Draw reference diagonal indicating full capacity boundary
    capacities = sorted(df["screen_capacity"].unique())
    axes[0].plot([0, 320], [0, 320], color="red", linestyle="--", label="100% Capacity Ceiling")
    axes[0].set_title("Tickets Sold vs Screen Capacity (Constraint Boundary)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Screen Capacity (Seats)")
    axes[0].set_ylabel("Tickets Sold")
    axes[0].legend(loc="upper left")
    
    sns.boxplot(data=df, x="screen_capacity", y="occupancy_pct", palette="viridis", ax=axes[1])
    axes[1].set_title("Occupancy % Distribution by Screen Capacity", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Screen Capacity (Seats)")
    axes[1].set_ylabel("Occupancy Rate (%)")
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "screen_capacity_vs_tickets.png")
    fig.savefig(fig5_path, dpi=200)
    plt.close()
    
    # 6. Correlation Heatmap
    num_cols = [
        "tickets_sold", "screen_capacity", "ticket_price", "days_since_release",
        "historical_movie_demand", "historical_weekday_demand", "historical_time_slot_demand"
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation Heatmap of Numerical Features & Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, "correlation_heatmap.png")
    fig.savefig(fig6_path, dpi=200)
    plt.close()
    
    # 7. Missing Values Analysis
    missing_series = df.isna().sum()
    missing_pct = (missing_series / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_series, "Missing %": missing_pct}).query("`Missing Count` > 0")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    if not missing_df.empty:
        sns.barplot(x=missing_df.index, y=missing_df["Missing Count"], palette="Reds_d", ax=ax)
        ax.set_title("Missing Values Count (Intentional Cold-Start Shows)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Missing Records")
        ax.set_xlabel("Feature")
        for i, v in enumerate(missing_df["Missing Count"]):
            ax.text(i, v + 0.5, f"{v} ({missing_df['Missing %'].iloc[i]:.2f}%)", ha="center", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center")
    plt.tight_layout()
    fig7_path = os.path.join(figures_dir, "missing_values_analysis.png")
    fig.savefig(fig7_path, dpi=200)
    plt.close()
    
    # 8. Outlier Analysis
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sns.boxplot(y=df["tickets_sold"], color="#9b59b6", ax=axes[0])
    axes[0].set_title("Outlier Check: tickets_sold", fontsize=12, fontweight="bold")
    
    sns.boxplot(y=df["ticket_price"], color="#3498db", ax=axes[1])
    axes[1].set_title("Outlier Check: ticket_price", fontsize=12, fontweight="bold")
    
    sns.boxplot(y=df["days_since_release"], color="#e67e22", ax=axes[2])
    axes[2].set_title("Outlier Check: days_since_release", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig8_path = os.path.join(figures_dir, "outlier_analysis.png")
    fig.savefig(fig8_path, dpi=200)
    plt.close()
    
    print(f"Generated all 8 EDA figures in {figures_dir}")
    
    # Now build the Jupyter Notebook JSON
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Cinema Ticket Demand Forecasting: Exploratory Data Analysis (EDA)\n",
                    "\n",
                    "## 1. Overview\n",
                    "This notebook presents an end-to-end exploratory analysis of the cinema show-level dataset. The dataset tracks individual shows across multiple screens and showtimes with strictly non-leaked historical demand features, calendar indicators, and operational ticket sales constraints."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "df = pd.read_csv('../data/raw/cinema_shows_raw.csv')\n",
                    "print(f'Total records: {len(df)}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Summary Statistics and Data Types"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.info()\n",
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Ticket Sales and Occupancy Distribution\n",
                    "Examines the target variable distribution and checks screen capacity boundaries."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots(figsize=(8, 4))\n",
                    "sns.histplot(df['tickets_sold'], kde=True, ax=ax, color='steelblue')\n",
                    "ax.set_title('Ticket Sales Distribution')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Weekday vs Weekend Demand Patterns\n",
                    "Cinema traffic surges on Friday, Saturday, and Sunday, with a moderate bump on Discount Tuesday."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
                    "plt.figure(figsize=(9, 4))\n",
                    "sns.barplot(data=df, x='weekday', y='tickets_sold', order=weekday_order, palette='crest')\n",
                    "plt.title('Mean Demand by Day of Week')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Showtime Demand\n",
                    "Evening (17:15) and Night (20:30) shows capture prime audience share, while Morning (10:30) shows show reduced attendance."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8, 4))\n",
                    "sns.boxplot(data=df, x='show_time', y='tickets_sold', palette='Set2')\n",
                    "plt.title('Tickets Sold Across Showtimes')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Correlation Heatmap\n",
                    "Historical movie demand and historical time slot demand show positive correlation with tickets sold, validating their predictive utility."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [],
                "source": [
                    "num_cols = ['tickets_sold', 'screen_capacity', 'ticket_price', 'days_since_release', 'historical_movie_demand', 'historical_weekday_demand', 'historical_time_slot_demand']\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')\n",
                    "plt.title('Feature Correlation Matrix')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Cold-Start and Missing Values Analysis\n",
                    "Missing values in `historical_movie_demand` represent cold-start shows (first show of a new release). These require explicit handling in the preprocessing and baseline fallback pipelines."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {},
                "outputs": [],
                "source": [
                    "missing = df.isna().sum()\n",
                    "print('Missing Values Count:\\n', missing[missing > 0])"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
        
    print(f"Jupyter Notebook successfully written to {notebook_path}")


if __name__ == "__main__":
    generate_eda_figures_and_notebook()

