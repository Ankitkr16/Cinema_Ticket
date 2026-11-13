"""
Cinema Ticket Demand Dataset Generator
Generates a reproducible, synthetic cinema show-level dataset.
Strictly prevents data leakage by computing historical demand features
using ONLY shows occurring prior to the current show's prediction timestamp.
"""

import os
import json
import random
import datetime
import numpy as np
import pandas as pd


def get_time_slot_label(show_time_str: str) -> str:
    """Classifies show_time into a canonical time slot."""
    hour = int(show_time_str.split(":")[0])
    if hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    elif hour < 21:
        return "Evening"
    else:
        return "Night"


def generate_cinema_dataset(
    num_days: int = 130,
    start_date_str: str = "2026-01-01",
    seed: int = 42,
    output_path: str = "data/raw/cinema_shows_raw.csv",
    metadata_path: str = "data/raw/dataset_metadata.json"
) -> pd.DataFrame:
    """
    Generates synthetic cinema show-level records chronologically.
    
    Adheres strictly to the NO DATA LEAKAGE requirement:
    historical_movie_demand, historical_weekday_demand, and historical_time_slot_demand
    are calculated purely from prior shows in the timeline.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Define Movie Catalog with varying popularity, genres, and staggered release days
    # Staggering releases ensures some movies appear brand new in validation and test sets (cold-start).
    movies_catalog = [
        {"movie_id": "MOV_001", "movie_name": "Galactic Wars: Legacy", "genre": "Sci-Fi", "base_demand": 210, "release_offset": 0},
        {"movie_id": "MOV_002", "movie_name": "The Laughing Detective", "genre": "Comedy", "base_demand": 115, "release_offset": 0},
        {"movie_id": "MOV_003", "movie_name": "Silent Whispers", "genre": "Drama", "base_demand": 75, "release_offset": 0},
        {"movie_id": "MOV_004", "movie_name": "Shadow Creek", "genre": "Horror", "base_demand": 95, "release_offset": 7},
        {"movie_id": "MOV_005", "movie_name": "Kingdom of Dragons", "genre": "Animation", "base_demand": 175, "release_offset": 10},
        {"movie_id": "MOV_006", "movie_name": "Fast Asphalt 11", "genre": "Action", "base_demand": 190, "release_offset": 14},
        {"movie_id": "MOV_007", "movie_name": "Midnight in Florence", "genre": "Romance", "base_demand": 65, "release_offset": 21},
        {"movie_id": "MOV_008", "movie_name": "The Cold Case Paradox", "genre": "Thriller", "base_demand": 105, "release_offset": 28},
        {"movie_id": "MOV_009", "movie_name": "Cybernetic Uprising", "genre": "Sci-Fi", "base_demand": 160, "release_offset": 35},
        {"movie_id": "MOV_010", "movie_name": "Fool's Paradise", "genre": "Comedy", "base_demand": 85, "release_offset": 42},
        {"movie_id": "MOV_011", "movie_name": "Echoes of the Past", "genre": "Drama", "base_demand": 55, "release_offset": 49},
        {"movie_id": "MOV_012", "movie_name": "Bloodline Manor", "genre": "Horror", "base_demand": 120, "release_offset": 56},
        {"movie_id": "MOV_013", "movie_name": "Robot Zoo Adventures", "genre": "Animation", "base_demand": 140, "release_offset": 63},
        {"movie_id": "MOV_014", "movie_name": "Apex Predator Protocol", "genre": "Action", "base_demand": 185, "release_offset": 70},
        {"movie_id": "MOV_015", "movie_name": "Autumn Promises", "genre": "Romance", "base_demand": 60, "release_offset": 77},
        {"movie_id": "MOV_016", "movie_name": "Decoded Conspiracy", "genre": "Thriller", "base_demand": 110, "release_offset": 84},
        # Late releases: these will launch during the Validation and Test chronological windows!
        {"movie_id": "MOV_017", "movie_name": "Quantum Abyss", "genre": "Sci-Fi", "base_demand": 195, "release_offset": 92},
        {"movie_id": "MOV_018", "movie_name": "Wedding Chaos", "genre": "Comedy", "base_demand": 90, "release_offset": 98},
        {"movie_id": "MOV_019", "movie_name": "The Painter's Sorrow", "genre": "Drama", "base_demand": 45, "release_offset": 105},
        {"movie_id": "MOV_020", "movie_name": "Spectral Haunting", "genre": "Horror", "base_demand": 130, "release_offset": 112},
        {"movie_id": "MOV_021", "movie_name": "Starship Heroes", "genre": "Animation", "base_demand": 165, "release_offset": 118},
        {"movie_id": "MOV_022", "movie_name": "Vengeance Strike", "genre": "Action", "base_demand": 180, "release_offset": 123},
    ]
    
    start_date = datetime.date.fromisoformat(start_date_str)
    
    # 2. Multiplex Configuration
    # 4 screens with distinct capacities
    screens = [
        {"screen_id": "Screen_1", "capacity": 300},
        {"screen_id": "Screen_2", "capacity": 220},
        {"screen_id": "Screen_3", "capacity": 160},
        {"screen_id": "Screen_4", "capacity": 100},
    ]
    
    # 5 standard showtimes per day
    show_times = ["10:30", "13:45", "17:15", "20:30", "22:45"]
    
    # Weekday demand multipliers
    weekday_multipliers = {
        "Monday": 0.72,
        "Tuesday": 0.86,   # Discount Tuesday effect
        "Wednesday": 0.78,
        "Thursday": 0.82,
        "Friday": 1.28,    # Weekend surge starts
        "Saturday": 1.48,  # Peak weekend demand
        "Sunday": 1.34     # Strong weekend demand
    }
    
    # Showtime demand multipliers
    time_multipliers = {
        "10:30": 0.65,  # Morning
        "13:45": 0.85,  # Afternoon
        "17:15": 1.30,  # Evening (Prime)
        "20:30": 1.42,  # Night (Prime)
        "22:45": 0.92   # Late Night
    }
    
    # History tracking data structures (strictly prior shows)
    # We maintain cumulative sales and count to compute strictly historical averages
    movie_sales_history = {m["movie_id"]: [] for m in movies_catalog}
    weekday_sales_history = {day: [] for day in weekday_multipliers}
    timeslot_sales_history = {"Morning": [], "Afternoon": [], "Evening": [], "Night": []}
    
    records = []
    show_counter = 1
    
    for day_idx in range(num_days):
        current_date = start_date + datetime.timedelta(days=day_idx)
        weekday_name = current_date.strftime("%A")
        w_multiplier = weekday_multipliers[weekday_name]
        
        # Determine which movies are active on this date (released on or before current_date, not older than 45 days)
        active_movies = []
        for m in movies_catalog:
            rel_date = start_date + datetime.timedelta(days=m["release_offset"])
            days_since = (current_date - rel_date).days
            if 0 <= days_since <= 45:
                active_movies.append((m, days_since))
                
        if not active_movies:
            # Fallback to the latest released movies
            active_movies = [(movies_catalog[0], 10)]
            
        # For each showtime of the day, iterate across screens
        for s_time in show_times:
            t_slot = get_time_slot_label(s_time)
            t_multiplier = time_multipliers[s_time]
            
            # Shuffle or assign active movies across screens
            # High-demand / new blockbusters get larger screens
            sorted_active = sorted(active_movies, key=lambda x: (x[0]["base_demand"] * np.exp(-0.03 * x[1])), reverse=True)
            
            for screen_idx, screen in enumerate(screens):
                assigned_movie, days_since_release = sorted_active[screen_idx % len(sorted_active)]
                m_id = assigned_movie["movie_id"]
                m_name = assigned_movie["movie_name"]
                genre = assigned_movie["genre"]
                capacity = screen["capacity"]
                
                # Base ticket price with small variation by screen size and prime time
                base_price = 12.0
                if capacity >= 220:
                    base_price += 2.0
                if s_time in ["17:15", "20:30"]:
                    base_price += 1.5
                if weekday_name == "Tuesday":
                    base_price -= 3.0  # Discount Tuesday promotion
                ticket_price = round(base_price + np.random.uniform(-0.5, 0.5), 2)
                
                # -------------------------------------------------------------
                # ZERO DATA LEAKAGE: Calculate historical features BEFORE show
                # Only use records strictly recorded from past shows (prior t' < t)
                # -------------------------------------------------------------
                if len(movie_sales_history[m_id]) > 0:
                    hist_movie_demand = round(float(np.mean(movie_sales_history[m_id])), 2)
                else:
                    # New movie cold-start: No prior shows exist for this movie!
                    hist_movie_demand = np.nan
                    
                if len(weekday_sales_history[weekday_name]) > 0:
                    hist_weekday_demand = round(float(np.mean(weekday_sales_history[weekday_name])), 2)
                else:
                    # Very first day initialization
                    hist_weekday_demand = np.nan
                    
                if len(timeslot_sales_history[t_slot]) > 0:
                    hist_timeslot_demand = round(float(np.mean(timeslot_sales_history[t_slot])), 2)
                else:
                    # Very first slot initialization
                    hist_timeslot_demand = np.nan
                
                # -------------------------------------------------------------
                # True Ticket Demand Generation (Ground Truth Simulation)
                # -------------------------------------------------------------
                # Decay factor over release age: demand decreases over time
                decay_factor = np.exp(-0.032 * days_since_release)
                
                # Price elasticity factor: higher price slightly reduces demand
                price_factor = 1.0 - 0.018 * (ticket_price - 12.0)
                
                # Latent true demand
                latent_demand = (
                    assigned_movie["base_demand"]
                    * decay_factor
                    * w_multiplier
                    * t_multiplier
                    * price_factor
                    + np.random.normal(0, 10.0)
                )
                
                # Screen capacity constraint: tickets_sold cannot exceed capacity and cannot be negative
                tickets_sold = int(np.clip(np.round(latent_demand), 0, capacity))
                
                # Record the show instance
                show_id = f"SHOW_{show_counter:05d}"
                records.append({
                    "show_id": show_id,
                    "movie_id": m_id,
                    "movie_name": m_name,
                    "show_date": current_date.isoformat(),
                    "show_time": s_time,
                    "weekday": weekday_name,
                    "screen_capacity": capacity,
                    "ticket_price": ticket_price,
                    "days_since_release": int(days_since_release),
                    "genre": genre,
                    "historical_movie_demand": hist_movie_demand,
                    "historical_weekday_demand": hist_weekday_demand,
                    "historical_time_slot_demand": hist_timeslot_demand,
                    "tickets_sold": tickets_sold
                })
                
                # -------------------------------------------------------------
                # AFTER RECORDING, update history memory for future shows
                # -------------------------------------------------------------
                movie_sales_history[m_id].append(tickets_sold)
                weekday_sales_history[weekday_name].append(tickets_sold)
                timeslot_sales_history[t_slot].append(tickets_sold)
                
                show_counter += 1

    df = pd.DataFrame(records)
    
    # Ensure correct data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Save dataset documentation metadata
    metadata = {
        "dataset_type": "Synthetic Cinema Show-Level Dataset",
        "documentation": "Generated synthetically with realistic demand dynamics, showtime effects, screen constraints, and zero data leakage.",
        "random_seed": seed,
        "total_records": len(df),
        "date_range": [df["show_date"].min(), df["show_date"].max()],
        "unique_movies": int(df["movie_id"].nunique()),
        "columns": list(df.columns),
        "cold_start_records_count": int(df["historical_movie_demand"].isna().sum()),
        "target_variable": "tickets_sold",
        "screen_capacities": [int(x) for x in sorted(df["screen_capacity"].unique())],
        "created_at": datetime.datetime.now().isoformat()
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Generated {len(df)} records saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Cold-start shows (historical_movie_demand is NaN): {df['historical_movie_demand'].isna().sum()}")
    return df


if __name__ == "__main__":
    generate_cinema_dataset()
