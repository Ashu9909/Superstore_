
#  Superstore Dashboard  —  Flask API

#    GET  /                        → serves the dashboard HTML
#    GET  /api/summary             → KPI cards
#    GET  /api/sales-trend         → monthly sales time-series
#    GET  /api/category            → sales + profit by category
#    GET  /api/region              → sales + margin by region
#    GET  /api/discount-impact     → avg profit per discount bucket
#    GET  /api/top-products        → top 10 sub-categories by sales
#    POST /api/predict-demand      → demand forecast (next N days)
#    POST /api/predict-price       → optimal price for a category


from flask import Flask, jsonify, request, render_template, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

#  DATA LOADING


BASE_DIR   = Path(__file__).parent
DATA_PATHS = [
    BASE_DIR / "superstore_cleaned.csv",
    BASE_DIR / "Sample - Superstore.csv",
    BASE_DIR / "superstore.csv",
]

def _load_raw() -> pd.DataFrame:
    for p in DATA_PATHS:
        if p.exists():
            df = pd.read_csv(p, encoding="latin-1")
            # normalise column names
            df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
            return df
    raise FileNotFoundError(
        "No dataset found. Place 'Sample - Superstore.csv' or 'superstore_cleaned.csv' "
        "in the same folder as app.py"
    )

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # standardise key column names across variants
    renames = {
        "order_date":  "order_date",
        "ship_date":   "ship_date",
        "ship_mode":   "ship_mode",
        "sub_category":"sub_category",
        "sub-category":"sub_category",
    }
    df = df.rename(columns=renames)

    needed = ["order_date", "sales", "quantity", "discount", "profit",
              "category", "region", "segment"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["order_date", "sales", "quantity"])
    df = df[(df["sales"] > 0) & (df["quantity"] > 0)]
    df = df.drop_duplicates()

    # derived fields
    df["price_per_unit"]    = (df["sales"] / df["quantity"]).round(2)
    df["profit_margin_pct"] = ((df["profit"] / df["sales"]) * 100).round(2)
    df["month"]             = df["order_date"].dt.month
    df["year"]              = df["order_date"].dt.year
    df["quarter"]           = df["order_date"].dt.quarter
    df["day_of_week"]       = df["order_date"].dt.dayofweek
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)

    if "ship_date" in df.columns:
        df["ship_date"]      = pd.to_datetime(df["ship_date"], errors="coerce")
        df["delivery_days"]  = (df["ship_date"] - df["order_date"]).dt.days

    df["discount_bucket"] = pd.cut(
        df["discount"],
        bins=[-0.01, 0.0, 0.1, 0.2, 0.5, 1.0],
        labels=["No Discount", "Low ≤10%", "Mid ≤20%", "High ≤50%", "Very High"]
    )
    return df

#  Eager load 
try:
    DF = _clean(_load_raw())
    print(f"✅ Dataset loaded: {len(DF):,} rows")
except Exception as e:
    print(f"⚠️  Dataset not found — using synthetic demo data. ({e})")
    #  Synthetic demo data so the API still works without the CSV 
    np.random.seed(42)
    n = 2000
    cats   = ["Technology", "Furniture", "Office Supplies"]
    regions= ["West", "East", "Central", "South"]
    segs   = ["Consumer", "Corporate", "Home Office"]
    dates  = pd.date_range("2020-01-01", periods=n, freq="D")[:n]
    dates  = np.random.choice(dates, n)
    DF = pd.DataFrame({
        "order_date"       : dates,
        "sales"            : np.random.lognormal(5, 1, n).round(2),
        "quantity"         : np.random.randint(1, 10, n),
        "discount"         : np.random.choice([0,0.1,0.2,0.3,0.4,0.5], n),
        "profit"           : np.random.normal(50, 80, n).round(2),
        "category"         : np.random.choice(cats, n),
        "region"           : np.random.choice(regions, n),
        "segment"          : np.random.choice(segs, n),
        "sub_category"     : np.random.choice(
            ["Phones","Chairs","Binders","Storage","Tables","Copiers",
             "Bookcases","Appliances","Accessories","Machines"], n),
    })
    DF = _clean(DF)


#  ML MODELS  (train once at startup)

def _build_forecast_model(df):
    """Lag-feature regression demand forecasting model."""
    daily = (df.groupby("order_date")["sales"]
               .sum().reset_index().sort_values("order_date"))
    daily.columns = ["ds", "y"]
    daily = daily.set_index("ds").asfreq("D").fillna(0).reset_index()

    for lag in [1, 7, 14, 28]:
        daily[f"lag_{lag}"] = daily["y"].shift(lag)
    daily["roll7_mean"]  = daily["y"].shift(1).rolling(7).mean()
    daily["roll14_mean"] = daily["y"].shift(1).rolling(14).mean()
    daily["roll7_std"]   = daily["y"].shift(1).rolling(7).std()
    daily["month"]       = daily["ds"].dt.month
    daily["dow"]         = daily["ds"].dt.dayofweek
    daily["quarter"]     = daily["ds"].dt.quarter
    daily["is_weekend"]  = (daily["dow"] >= 5).astype(int)
    daily = daily.dropna()

    feats = [c for c in daily.columns if c not in ["ds","y"]]
    split = daily["ds"].quantile(0.8)
    tr    = daily[daily["ds"] <= split]
    te    = daily[daily["ds"] >  split]

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42)
    model.fit(tr[feats], tr["y"])
    preds = np.clip(model.predict(te[feats]), 0, None)
    mae   = mean_absolute_error(te["y"], preds)
    r2    = r2_score(te["y"], preds)
    print(f"   Forecast model  MAE={mae:.1f}  R²={r2:.3f}")
    return model, feats, daily

def _build_price_model(df):
    """Price–demand model with categorical features."""
    le_cat = LabelEncoder().fit(df["category"])
    le_reg = LabelEncoder().fit(df["region"])
    le_seg = LabelEncoder().fit(df["segment"])

    X = pd.DataFrame({
        "price"   : df["price_per_unit"],
        "discount": df["discount"],
        "cat_enc" : le_cat.transform(df["category"]),
        "reg_enc" : le_reg.transform(df["region"]),
        "seg_enc" : le_seg.transform(df["segment"]),
    })
    y = df["quantity"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    r2  = r2_score(y_te, model.predict(X_te))
    print(f"   Price model     R²={r2:.3f}")
    return model, le_cat, le_reg, le_seg

print("Training ML models …")
FORECAST_MODEL, FORECAST_FEATS, DAILY_DF = _build_forecast_model(DF)
PRICE_MODEL, LE_CAT, LE_REG, LE_SEG      = _build_price_model(DF)
print("✅ Models ready.")

#  HELPER

def _safe(val):
    """Convert numpy scalars to Python natives for JSON."""
    if isinstance(val, (np.integer,)): return int(val)
    if isinstance(val, (np.floating,)): return round(float(val), 2)
    return val

#  ROUTES  —  Static

@app.route("/")
def index():
    return render_template("dashboard.html")

#  ROUTES  —  Analytics API

@app.route("/api/summary")
def api_summary():
    d = DF
    churn_risk = int((d["profit_margin_pct"] < 0).sum())
    return jsonify({
        "total_sales"      : _safe(d["sales"].sum()),
        "total_profit"     : _safe(d["profit"].sum()),
        "total_orders"     : int(len(d)),
        "avg_margin_pct"   : _safe(d["profit_margin_pct"].mean()),
        "avg_order_value"  : _safe(d["sales"].mean()),
        "loss_making_rows" : churn_risk,
        "categories"       : int(d["category"].nunique()),
        "regions"          : int(d["region"].nunique()),
    })

@app.route("/api/sales-trend")
def api_sales_trend():
    monthly = (DF.groupby(DF["order_date"].dt.to_period("M"))["sales"]
                 .sum().reset_index())
    monthly["order_date"] = monthly["order_date"].dt.to_timestamp()
    monthly = monthly.sort_values("order_date")
    return jsonify({
        "labels": monthly["order_date"].dt.strftime("%b %Y").tolist(),
        "values": [_safe(v) for v in monthly["sales"].tolist()],
    })

@app.route("/api/category")
def api_category():
    g = DF.groupby("category").agg(
        sales=("sales","sum"), profit=("profit","sum"),
        orders=("sales","count")
    ).reset_index()
    g["margin"] = (g["profit"] / g["sales"] * 100).round(2)
    return jsonify(g.apply(lambda r: {
        "category": r["category"],
        "sales"   : _safe(r["sales"]),
        "profit"  : _safe(r["profit"]),
        "orders"  : int(r["orders"]),
        "margin"  : _safe(r["margin"]),
    }, axis=1).tolist())

@app.route("/api/region")
def api_region():
    g = DF.groupby("region").agg(
        sales=("sales","sum"), profit=("profit","sum"),
        orders=("sales","count")
    ).reset_index()
    g["margin"] = (g["profit"] / g["sales"] * 100).round(2)
    g = g.sort_values("sales", ascending=False)
    return jsonify(g.apply(lambda r: {
        "region" : r["region"],
        "sales"  : _safe(r["sales"]),
        "profit" : _safe(r["profit"]),
        "orders" : int(r["orders"]),
        "margin" : _safe(r["margin"]),
    }, axis=1).tolist())

@app.route("/api/discount-impact")
def api_discount_impact():
    g = DF.groupby("discount_bucket", observed=True).agg(
        avg_profit=("profit","mean"),
        avg_sales =("sales","mean"),
        count     =("sales","count"),
    ).reset_index()
    return jsonify(g.apply(lambda r: {
        "bucket"     : str(r["discount_bucket"]),
        "avg_profit" : _safe(r["avg_profit"]),
        "avg_sales"  : _safe(r["avg_sales"]),
        "count"      : int(r["count"]),
    }, axis=1).tolist())

@app.route("/api/top-products")
def api_top_products():
    col = "sub_category" if "sub_category" in DF.columns else "category"
    g = (DF.groupby(col)["sales"].sum()
           .nlargest(10).reset_index()
           .sort_values("sales"))
    return jsonify({
        "labels": g[col].tolist(),
        "values": [_safe(v) for v in g["sales"].tolist()],
    })

@app.route("/api/segment")
def api_segment():
    g = DF.groupby("segment").agg(
        sales=("sales","sum"), profit=("profit","sum")
    ).reset_index()
    return jsonify(g.apply(lambda r: {
        "segment": r["segment"],
        "sales"  : _safe(r["sales"]),
        "profit" : _safe(r["profit"]),
    }, axis=1).tolist())


#  ROUTES  —  ML Prediction API

@app.route("/api/predict-demand", methods=["POST"])
def api_predict_demand():
    """
    Body: { "days": 30 }
    Returns forecast for next N days.
    """
    body    = request.get_json(silent=True) or {}
    n_days  = min(int(body.get("days", 30)), 180)

    last    = DAILY_DF.iloc[-1].copy()
    last_ds = DAILY_DF["ds"].max()
    preds   = []
    history = DAILY_DF["y"].values.copy()

    for i in range(1, n_days + 1):
        next_ds = last_ds + pd.Timedelta(days=i)
        row = {
            "lag_1"       : history[-1],
            "lag_7"       : history[-7]  if len(history) >= 7  else history[0],
            "lag_14"      : history[-14] if len(history) >= 14 else history[0],
            "lag_28"      : history[-28] if len(history) >= 28 else history[0],
            "roll7_mean"  : float(np.mean(history[-7:])),
            "roll14_mean" : float(np.mean(history[-14:])),
            "roll7_std"   : float(np.std(history[-7:])),
            "month"       : next_ds.month,
            "dow"         : next_ds.dayofweek,
            "quarter"     : next_ds.quarter,
            "is_weekend"  : int(next_ds.dayofweek >= 5),
        }
        X_row  = pd.DataFrame([row])[FORECAST_FEATS]
        y_hat  = float(np.clip(FORECAST_MODEL.predict(X_row)[0], 0, None))
        preds.append({"date": next_ds.strftime("%Y-%m-%d"), "predicted_sales": round(y_hat, 2)})
        history = np.append(history, y_hat)

    return jsonify({
        "days"       : n_days,
        "forecast"   : preds,
        "total_predicted": round(sum(p["predicted_sales"] for p in preds), 2),
    })


@app.route("/api/predict-price", methods=["POST"])
def api_predict_price():
    """
    Body: { "category": "Technology", "region": "West",
            "segment": "Consumer", "discount": 0.1 }
    Returns optimal price and revenue curve.
    """
    body     = request.get_json(silent=True) or {}
    category = body.get("category", DF["category"].mode()[0])
    region   = body.get("region",   DF["region"].mode()[0])
    segment  = body.get("segment",  DF["segment"].mode()[0])
    discount = float(body.get("discount", DF["discount"].median()))

    # Validate inputs
    valid_cats = DF["category"].unique().tolist()
    if category not in valid_cats:
        return jsonify({"error": f"Unknown category. Valid: {valid_cats}"}), 400

    try:
        cat_enc = int(LE_CAT.transform([category])[0])
        reg_enc = int(LE_REG.transform([region])[0])
        seg_enc = int(LE_SEG.transform([segment])[0])
    except Exception:
        return jsonify({"error": "Unknown region or segment value"}), 400

    cat_df    = DF[DF["category"] == category]
    price_min = float(cat_df["price_per_unit"].quantile(0.05))
    price_max = float(cat_df["price_per_unit"].quantile(0.95))
    prices    = np.linspace(price_min, price_max, 100)

    sim = pd.DataFrame({
        "price"   : prices,
        "discount": discount,
        "cat_enc" : cat_enc,
        "reg_enc" : reg_enc,
        "seg_enc" : seg_enc,
    })
    qty     = np.clip(PRICE_MODEL.predict(sim), 0, None)
    revenue = prices * qty
    opt_idx = int(np.argmax(revenue))

    curve = [{"price": round(float(p),2), "revenue": round(float(r),2)}
             for p, r in zip(prices[::5], revenue[::5])]   # thin to 20 pts

    return jsonify({
        "category"          : category,
        "optimal_price"     : round(float(prices[opt_idx]), 2),
        "expected_quantity" : round(float(qty[opt_idx]), 2),
        "expected_revenue"  : round(float(revenue[opt_idx]), 2),
        "price_range"       : [round(price_min,2), round(price_max,2)],
        "revenue_curve"     : curve,
    })


#  Health check

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "rows": len(DF)})

if __name__ == "__main__":
    print("\n🚀  Starting Superstore Dashboard")
    print("   Open → http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
