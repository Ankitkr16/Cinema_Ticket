# Cinema Ticket Demand Forecasting: Technical Evaluation Report

**Author:** Machine Learning Engineering Team  
**Date:** 2026-11-13  
**Status:** Completed & Validated  
**Target Variable:** `tickets_sold` (Integer count bounded by `[0, screen_capacity]`)  

---

## Executive Summary

This report documents the design, implementation, and empirical validation of a **Cinema Ticket Demand Forecasting System** engineered to predict individual show-level ticket sales. The primary objective is to empower multiplex management to accurately anticipate seat occupancy, optimize screen allocation, align concessions inventory, and balance operational staffing.

The solution was developed with strict adherence to temporal sequence boundaries (zero future data leakage), evaluated across chronological splits, and packaged into an inference-ready FastAPI service and Streamlit operational dashboard.

---

## 1. Business Problem

Multiplex cinemas face complex operational scheduling decisions under fluctuating consumer demand:
- **Screen Allocation Inefficiencies:** Assigning high-demand blockbusters to smaller screens (e.g., 100 seats) leads to truncated sales and rejected revenue, while assigning low-demand films to large screens (e.g., 300 seats) incurs excessive heating, power, and cleaning overhead.
- **Staffing Bottlenecks:** Unpredictable prime-time surges overwhelm ticket scanning, concession stands, and usher cleaning teams.
- **Asymmetric Revenue Risk:** Box office revenue is perishable; an empty seat at showtime is lost revenue that cannot be recovered.

A reliable, explainable machine learning model that forecasts individual show ticket sales and expected occupancy allows management to make data-driven screen reassignments 24 to 72 hours before showtime.

---

## 2. Prediction Unit and Target

- **Prediction Unit:** An **individual cinema show event** scheduled on a specific screen at a specific date and time for a specific movie.
- **Target Variable (`tickets_sold`):** Integer count of tickets sold for that specific show.
- **Physical & Business Constraints:**
  $$\forall i, \quad 0 \le \hat{y}_i \le \text{screen\_capacity}_i$$
  Predicted ticket demand must never be negative and can never exceed the physical capacity of the allocated auditorium.

---

## 3. Forecast Horizon and Prediction Time

- **Prediction Time:** Defined strictly as **before the show commences** (e.g., 24 to 48 hours prior to showtime during weekly scheduling).
- **Forecast Horizon:** Next scheduled showtime up to 7 days ahead.
- **Temporal Information Cutoff:** All historical statistics, past movie performance, and day-of-week averages available to the model must have occurred strictly prior to the show's timestamp ($t' < t_{\text{show}}$).

---

## 4. Dataset Description and Source

To enable rigorous, fully reproducible experimentation without commercial licensing restrictions, a synthetic dataset generator was engineered (`src/data_generator.py`) using fixed random seed (`seed=42`).

- **Total Records:** 2,600 individual cinema shows.
- **Date Range:** 2026-01-01 to 2026-05-10 (130 consecutive calendar days).
- **Screens:** 4 distinct auditorium screens with capacities of 100, 160, 220, and 300 seats.
- **Showtimes:** 5 standardized daily time slots (10:30, 13:45, 17:15, 20:30, 22:45).
- **Movie Catalog:** 22 movies across 8 genres (Action, Sci-Fi, Comedy, Drama, Horror, Animation, Romance, Thriller) spanning varied popularity tiers (Blockbuster, Hit, Moderate, Niche, Flop) and staggered release offsets.

### Dataset Schema
| Column Name | Data Type | Description |
|---|---|---|
| `show_id` | String | Unique show identifier (e.g., `SHOW_00001`) |
| `movie_id` | String | Unique movie identifier (e.g., `MOV_001`) |
| `movie_name` | String | Movie title |
| `show_date` | Date | Date of show (`YYYY-MM-DD`) |
| `show_time` | String | Scheduled start time (`HH:MM`) |
| `weekday` | String | Day of week (`Monday` through `Sunday`) |
| `screen_capacity`| Integer | Auditorium seat capacity (100, 160, 220, 300) |
| `ticket_price` | Float | Base admission price in currency units ($8.50 - $17.50) |
| `days_since_release`| Integer | Days elapsed since theatrical release date |
| `genre` | String | Thematic category |
| `historical_movie_demand` | Float | Cumulative average tickets sold for this movie across strictly prior shows |
| `historical_weekday_demand` | Float | Cumulative average tickets sold on this weekday across strictly prior shows |
| `historical_time_slot_demand` | Float | Cumulative average tickets sold in this slot across strictly prior shows |
| `tickets_sold` | Integer | Actual tickets sold (Target) |

---

## 5. Exploratory Data Analysis (EDA) Findings

Comprehensive EDA was conducted and visual artifacts were saved to `reports/figures/`:

1. **Ticket Sales Distribution (`ticket_sales_distribution.png`):** Shows exhibit a multi-modal distribution reflecting screen capacities and popularity tiers, ranging from 12 tickets for late-run weekday dramas up to 300 (sold-out) for weekend blockbusters.
2. **Day-of-Week Effect (`weekday_vs_weekend_demand.png`):** Weekend shows (Friday through Sunday) average ~128 tickets/show compared to ~78 tickets/show on weekdays (Monday through Thursday), with a noticeable secondary bump on "Discount Tuesdays".
3. **Showtime Dynamics (`showtime_demand.png`):** Peak demand occurs at 20:30 (Night Prime) and 17:15 (Evening Prime), while morning shows (10:30) exhibit lower occupancy (~40-55%).
4. **Capacity Truncation (`screen_capacity_vs_tickets.png`):** A strong ceiling effect is visible at each screen tier (100, 160, 220, 300), confirming that unconstrained latent demand is frequently clipped by auditorium size.
5. **Feature Correlation (`correlation_heatmap.png`):** Historical movie demand ($r \approx 0.62$) and historical time slot demand ($r \approx 0.48$) have the strongest positive correlation with tickets sold, while days since release ($r \approx -0.41$) demonstrates exponential decay.
6. **Missing Values Analysis (`missing_values_analysis.png`):** 20 records in `historical_movie_demand` have missing values ($NaN$). These correspond to the opening shows of new releases where zero prior show records exist.

---

## 6. Data Preprocessing

Data preprocessing is encapsulated in a reproducible Scikit-Learn `ColumnTransformer` (`src/preprocessing.py`):
1. **Missing Numerical Imputation:** `SimpleImputer(strategy='median', add_indicator=True)` imputes missing historical values using training medians while preserving a binary missingness indicator column (`is_cold_start`).
2. **Numerical Scaling:** `StandardScaler` standardizes numerical features (`screen_capacity`, `ticket_price`, `days_since_release`, `historical_movie_demand`, `historical_weekday_demand`, `historical_time_slot_demand`, `show_hour`, `is_weekend`).
3. **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` transforms `genre`, `weekday`, and `time_slot`.
4. **Data Type Validation:** Rejects non-ISO dates, negative capacities, and out-of-range prices.

---

## 7. Feature Engineering

Domain-specific features extracted prior to model training include:
- `is_weekend`: Binary flag for Friday, Saturday, and Sunday.
- `time_slot`: Categorical binning into Morning (<12:00), Afternoon (12:00-16:59), Evening (17:00-20:59), and Night (>=21:00).
- `show_hour`: Extracted integer hour of screening.
- `is_cold_start`: Indicator flag identifying shows with zero prior movie history.

---

## 8. Data Leakage Prevention

Data leakage is a critical risk in temporal forecasting. The following safeguards were enforced:
1. **Strict Prior-Time Window:** During dataset generation, show $t$ was generated using historical averages calculated *only* over shows $t' < t$. The target `tickets_sold` of show $t$ was never accessible during feature construction.
2. **Chronological Splitting:** Random train/test shuffling was prohibited. Splits were carved strictly across chronological time boundaries.
3. **Imputer and Scaler Leakage Elimination:** The `ColumnTransformer` was fitted *exclusively* on the training set (earliest 70% of records) and only applied to validation and test records via `.transform()`.

---

## 9. Baseline Methodology

A domain-specific **Capacity-Adjusted Historical Baseline** (`src/baseline.py`) was constructed to represent standard heuristic cinema management practice:
- Computes empirical training set priors: $\mu_{\text{global}}$, movie-specific average $\mu_{\text{movie}}$, genre average $\mu_{\text{genre}}$, weekday multiplier $F_{\text{weekday}}$, and time-slot multiplier $F_{\text{slot}}$.
- For an upcoming show:
  $$\hat{y}_{\text{raw}} = \mu_{\text{base}} \times F_{\text{weekday}} \times F_{\text{slot}}$$
  where $\mu_{\text{base}} = \text{historical\_movie\_demand}$ if known, else $\mu_{\text{movie}}$ if seen in training, else genre fallback $\mu_{\text{genre}}$.
- Strictly applies capacity clamping:
  $$\hat{y}_{\text{baseline}} = \min(\max(0, \hat{y}_{\text{raw}}), \text{screen\_capacity})$$

---

## 10. Machine Learning Models

Two non-linear tree-based regression algorithms were trained and compared:

### 1. RandomForestRegressor
- Ensemble of 160 de-correlated decision trees with `max_depth=12`, `min_samples_split=4`.
- Averages predictions across bootstrap samples, reducing variance and resisting overfitting.

### 2. GradientBoostingRegressor
- Sequential boosting ensemble of 160 shallow trees with `learning_rate=0.07`, `max_depth=5`, `min_samples_split=5`.
- Minimizes squared error loss iteratively, capturing subtle interactions between showtimes, day of week, and release age decay.

Both models were integrated inside `CinemaDemandPredictor` (`src/models.py`), which automatically clamps raw model predictions to $[0, \text{screen\_capacity}]$.

---

## 11. Chronological Splitting Strategy

The dataset of 2,600 shows was sorted chronologically by date, time, and show ID:
- **Training Set (70%):** 1,819 shows (2026-01-01 to 2026-04-01)
- **Validation Set (15%):** 391 shows (2026-04-01 to 2026-04-21) — *Used strictly for hyperparameter tuning and model selection*
- **Test Set (15%):** 390 shows (2026-04-21 to 2026-05-10) — *Held out untouched until final model evaluation*

---

## 12. Empirical Model Evaluation Metrics

Performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Weighted Absolute Percentage Error (WAPE):

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

$$\text{WAPE} = \frac{\sum_{i=1}^{N}|y_i - \hat{y}_i|}{\sum_{i=1}^{N} y_i} \times 100\%$$

### Validation Set Results (Model Selection)
| Model Architecture | Validation MAE | Validation RMSE | Validation WAPE (%) | Selected |
|---|---|---|---|---|
| Capacity Baseline | 33.36 | 38.61 | 36.44% | No |
| RandomForestRegressor | 12.40 | 16.65 | 13.55% | No |
| **GradientBoostingRegressor** | **10.95** | **14.18** | **11.96%** | **Yes (Best)** |

Based on validation set WAPE and MAE, **Gradient Boosting** was selected as the production model artifact.

---

## 13. Final Test Set Model Comparison

The final evaluation was executed on the 390 untouched chronological test records:

| Model | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Capacity Violations |
|---|---|---|---|---|
| **Capacity Baseline** | 31.35 | 37.53 | 32.19% | 0 |
| **RandomForestRegressor** | 14.75 | 19.51 | 15.15% | 0 |
| **GradientBoostingRegressor** | **12.29** | **15.88** | **12.62%** | **0** |

### Key Observations:
1. **Substantial Baseline Outperformance:** Gradient Boosting cut the baseline MAE from 31.35 to 12.29 tickets (a **60.8% error reduction**) and WAPE from 32.19% to 12.62%.
2. **Zero Capacity Violations:** Unconstrained regression algorithms frequently predict values above auditorium capacity or negative tickets. With our capacity clamping layer, capacity violation count is **0**.

---

## 14. Cold-Start Analysis

In the cinema domain, new movies enter theaters without prior ticket sales history. To evaluate model robustness under cold-start conditions, the test set was segmented into three distinct maturity tiers:

| Maturity Segment | Definition | Shows ($n$) | Baseline MAE | Best Model MAE | Best Model WAPE (%) |
|---|---|---|---|---|---|
| **Unseen / New Movies** | Brand new release or zero prior shows in training | 344 | 31.60 | **12.67** | 12.17% |
| **Low-Data Movies** | $\le 40$ historical shows or release age $\le 7$ days | 18 | 29.83 | **10.25** | 25.44% |
| **Established Movies** | Extensive historical records and mature run | 28 | 29.28 | **9.00** | 17.35% |

### Analysis:
- **Error Gradient:** Model error monotonically increases as movie history decreases: **9.00 MAE (Established) $\rightarrow$ 10.25 MAE (Low-Data) $\rightarrow$ 12.67 MAE (New Movies)**.
- **Why New Movies Have Greater Uncertainty:** Before box office reception is established, audience interest relies heavily on marketing buzz and star cast appeal. The model relies on genre-level priors and time slot effects, which naturally introduce higher variance.
- **Operational Recommendation:** For opening weekend screenings, cinema managers should incorporate manual human review and maintain a 15-20% buffer when committing large screens.

---

## 15. Error Analysis and Diagnostics

Diagnostic charts (`reports/figures/error_distribution.png` and `reports/figures/actual_vs_predicted.png`) reveal:
1. **Residual Distribution:** Prediction residuals ($y - \hat{y}$) are symmetrically centered around zero with minimal bias (Mean Error $\approx -0.4$ tickets).
2. **Homoscedasticity:** Residual variance remains relatively stable across low, medium, and high predicted ticket counts, with slight expansion at full capacity (sold-out shows) due to demand truncation.

---

## 16. Limitations and Responsible AI Considerations

1. **Synthetic Nature:** While synthetically modeled after real box office dynamics, external disruptions (extreme weather, local transit strikes, sudden viral trends) are not represented in the data distribution.
2. **Cold-Start Uncertainty:** The system cannot guarantee pinpoint accuracy for unprecedented blockbuster phenomena or unpredicted flops during their first 48 hours.
3. **No Automated Overrides:** The model must serve as decision support. Final scheduling and ticket pricing adjustments must remain under managerial human oversight.

---

## 17. Conclusion

The Cinema Ticket Demand Forecasting System demonstrates high predictive fidelity, robust zero-leakage preprocessing, and strict capacity bounding. With a held-out test WAPE of **12.62%** (down from 32.19% baseline), the model provides cinema operations with a reliable tool for auditorium scheduling and staffing efficiency.

