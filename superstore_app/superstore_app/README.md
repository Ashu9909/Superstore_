# Superstore ML Dashboard — Flask API

End-to-end ML-powered analytics dashboard with REST API.

## Project Structure

```
superstore_app/
├── app.py                      # Flask API (all endpoints + ML models)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── templates/
│   └── dashboard.html          # Full analytics dashboard UI
└── Sample - Superstore.csv     # ← place your dataset here
```

## Setup (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Download from Kaggle and place in this folder:
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Accepted filenames (any of these works):
- `Sample - Superstore.csv`
- `superstore_cleaned.csv`
- `superstore.csv`

> If no dataset is found, the app runs on **synthetic demo data** automatically.

### 3. Run
```bash
python app.py
```

Open → http://127.0.0.1:5000

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/summary` | KPI cards — total sales, profit, orders, margin |
| GET | `/api/sales-trend` | Monthly revenue time-series |
| GET | `/api/category` | Sales + profit by product category |
| GET | `/api/region` | Regional performance + margin |
| GET | `/api/discount-impact` | Avg profit per discount bucket |
| GET | `/api/top-products` | Top 10 sub-categories by revenue |
| GET | `/api/segment` | Revenue by customer segment |
| POST | `/api/predict-demand` | ML demand forecast |
| POST | `/api/predict-price` | Price optimization |
| GET | `/api/health` | API health check |

---

## ML Endpoints — Request / Response

### POST /api/predict-demand
```json
// Request
{ "days": 30 }

// Response
{
  "days": 30,
  "total_predicted": 48500.25,
  "forecast": [
    { "date": "2024-01-01", "predicted_sales": 1620.50 },
    ...
  ]
}
```

### POST /api/predict-price
```json
// Request
{
  "category": "Technology",
  "region": "West",
  "segment": "Consumer",
  "discount": 0.1
}

// Response
{
  "category": "Technology",
  "optimal_price": 284.50,
  "expected_quantity": 3.2,
  "expected_revenue": 910.40,
  "price_range": [12.50, 1800.00],
  "revenue_curve": [
    { "price": 12.50, "revenue": 42.00 },
    ...
  ]
}
```

---

## ML Models Used

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| Demand Forecast | Predict next N days sales | GradientBoostingRegressor + lag features |
| Price Optimizer | Find revenue-maximizing price | RandomForestRegressor + simulation |

Both models train automatically at startup — no pre-trained files needed.

---

## Production Deployment (optional)

```bash
# With gunicorn (Linux/Mac)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# With waitress (Windows)
pip install waitress
waitress-serve --port=5000 app:app
```

---

## Dashboard Sections

- **Overview** — KPI cards + revenue trend + category/segment charts
- **Sales Trends** — time-series + discount impact + regional bar
- **Products** — top sub-categories + full category breakdown table
- **Regions** — regional sales + margin + status table
- **Demand Forecast** — interactive ML forecast with chart
- **Price Optimizer** — ML price simulation with revenue curve
- **API Reference** — all endpoints listed in-dashboard
