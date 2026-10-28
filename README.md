# Divar Real Estate Data Science Project

A complete Data Science workflow on approximately **1 million real estate listings from Divar**, covering preprocessing, EDA, hypothesis testing, clustering, and house price prediction.

## Project Overview

The goal is to clean and analyze real estate data and build machine learning models to discover market patterns and predict property prices.

**Key Results**
- Cleaned dataset: **999,997 rows × 79 features**
- Valid geographic coordinates: ~636,000
- Prediction dataset: 893,259 samples
- Final model: Random Forest Regressor
- Test \(R^2\): 0.6698 on the log-transformed target

## Workflow

```text
Raw Data → Preprocessing → EDA → Hypothesis Testing
                              ↓
                    Clustering → Prediction
```

## 1. Data Preprocessing

- Standardized numerical, categorical, Boolean, financial, text, and time features.
- Handled missing values based on their meaning, preserving structural missingness.
- Engineered new features and validated geographic coordinates.
- Preserved raw data where possible and investigated suspicious values.

**Output:** `Divar_cleaned.parquet`

## 2. EDA & Hypothesis Testing

Explored price and rent distributions, listing trends, geographic patterns, construction years, and property amenities.

Four main hypotheses investigated:
- Property sizes in metropolitan vs. smaller cities
- Property sizes by construction year
- Business deeds and commercial property prices
- Luxury amenities and villa prices

Welch's t-test and Mann–Whitney U test were used where appropriate. Associations were not interpreted as causation.

## 3. Clustering

Compared K-Means and DBSCAN using property prices, geographic coordinates, and selected property features.

- **K-Means:** KneeLocator suggested \(k=6\).
- **DBSCAN:** Sample of 50,000 listings, `eps=0.507`, `min_samples=20`.

| Metric | Result |
|---|---:|
| Silhouette Score | 0.473 |
| Davies–Bouldin Index | 0.890 |
| Noise Ratio | ~0.4% |

## 4. House Price Prediction

Built a regression pipeline for residential and commercial sales and rentals.

- Constructed a unified financial target.
- Removed suspicious target values using explicit rules.
- Applied target encoding and one-hot encoding.
- Addressed target leakage risks.
- Compared baseline, Linear Regression, Decision Tree, and Random Forest models.

**Final Model:** Random Forest Regressor with 100 trees and `min_samples_leaf=5`.

| Test Metric | Result |
|---|---:|
| MAE (log scale) | 0.4904 |
| RMSE (log scale) | 0.8433 |
| R² (log scale) | 0.6698 |

The model explains approximately 67% of the variance in the log-transformed target.

## Tech Stack

Python, Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn, Folium, PyArrow, Jupyter Notebook.

## Repository Structure

```text
divar-real-estate-project/
├── data/
├── notebook/
├── prediction/
├── scripts/
├── README.md
└── .gitignore
```

Large datasets are excluded from Git and must be provided locally.

## Getting Started

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow kneed jdatetime utm folium
```

1. Place `Divar.csv` in `data/`.
2. Run the main preprocessing notebook.
3. Use the cleaned Parquet dataset for subsequent analyses.

**Note:** This project is for educational and analytical purposes, not professional property valuation or financial advice.
