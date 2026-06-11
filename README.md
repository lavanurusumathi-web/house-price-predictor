# House Price Predictor

An end-to-end machine learning system that estimates house market values using 5 regression models. Built with Python, scikit-learn, and Flask — deployed on Vercel and Render.

**Live Demo:** [house-price-predictor.vercel.app](https://house-price-predictor.vercel.app) | [API](https://house-price-api.onrender.com)

---

## Features

- **Synthetic Data Generation** — Creates 1,200 realistic house records with derived prices using interaction terms and noise
- **Exploratory Data Analysis** — Statistical summaries and 5 visualization plots (distributions, correlations, feature relationships)
- **Model Training** — Trains and compares 5 regression models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
- **Future Price Forecasting** — Predicts 12-month price trends with confidence bands
- **Web Interface** — Dark-themed responsive UI with 12 input fields for real-time predictions
- **REST API** — Production-ready endpoints for Render and Vercel deployments

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11, JavaScript (vanilla) |
| ML/Analytics | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn |
| Serialization | joblib |
| Web Framework | Flask, Flask-CORS |
| Deployment | Vercel (serverless), Render (gunicorn) |

## Model Performance

The **Ridge (L2) Regression** model achieved the best results:

| Metric | Value |
|---|---|
| Test R² | 0.9724 |
| Test MAE | ~$39,492 |
| Test RMSE | ~$51,230 |

Other models tested: Linear Regression, Lasso, Random Forest, Gradient Boosting.

## Project Structure

```
├── main.py                  # Pipeline orchestrator
├── generate_data.py         # Synthetic data generator
├── explore_data.py          # EDA & visualization
├── train_model.py           # ML model training & comparison
├── predict_future.py        # Time-series future forecasting
├── app.py                   # Flask full-stack app (UI + API)
├── api.py                   # Render deployment API
├── api/
│   └── index.py             # Vercel serverless API
├── index.html               # Standalone frontend (Vercel)
├── data/
│   └── house_data.csv       # Generated dataset (1,200 rows × 14 cols)
├── models/
│   ├── ridge_(l2).pkl       # Best trained model
│   └── scaler.pkl           # Fitted StandardScaler
├── output/                  # Generated plots, CSVs, comparisons
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment config
├── vercel.json              # Vercel deployment config
└── .gitignore
```

## Pipeline Flow

```
generate_data.py  →  explore_data.py  →  train_model.py  →  predict_future.py
       ↓                    ↓                  ↓                    ↓
  data/house_data.csv    output/*.png    models/*.pkl         output/future_*.csv
```

The web apps (`app.py`, `api/index.py`) load the pre-trained Ridge model and scaler to serve real-time predictions.

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/lavanurusumathi-web/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
python main.py
```

### Run the Web App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## API Usage

**Endpoint:** `POST /api/predict`

**Request Body:**

```json
{
  "sqft": 1800,
  "bedrooms": 3,
  "bathrooms": 2.0,
  "year_built": 2005,
  "location_score": 7.5,
  "distance_to_city_miles": 5.0,
  "crime_rate": 3.0,
  "school_rating": 7.0,
  "has_garage": 1,
  "has_garden": 1,
  "floors": 2,
  "sale_year": 2025
}
```

**Response:**

```json
{
  "predicted_price": 875432.10,
  "currency": "USD"
}
```

## Deployment

### Vercel

The project is deployed at [house-price-predictor.vercel.app](https://house-price-predictor.vercel.app) with:
- Static frontend (`index.html`) at the root
- Serverless API at `/api/predict`

### Render

The API is also deployed at [house-price-api.onrender.com](https://house-price-api.onrender.com) using gunicorn with 2 workers.

## Input Features

| Feature | Range | Description |
|---|---|---|
| Square Footage | 600–5000 | Total living area |
| Bedrooms | 1–6 | Number of bedrooms |
| Bathrooms | 1.0–4.0 | Number of bathrooms |
| Year Built | 1950–2025 | Construction year |
| Location Score | 1–10 | Neighborhood quality rating |
| Distance to City | 0.5–40 mi | Proximity to urban center |
| Crime Rate | 0.5–10 | Local crime index |
| School Rating | 1–10 | Nearby school quality |
| Garage | Yes/No | Attached garage |
| Garden | Yes/No | Private garden/yard |
| Floors | 1–3 | Number of stories |
| Sale Year | 2020–2035 | Year of sale |
