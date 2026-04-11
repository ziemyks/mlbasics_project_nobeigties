"""
api.py — LightGBM Fraud Detection REST API
==========================================
Run:
    pip install flask flask-cors kagglehub
    python api.py

Then send a POST request:
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d @sample_transaction.json
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os, random, string, uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Load artefacts ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ART  = os.path.join(BASE, "model_artefacts")

model         = joblib.load(os.path.join(ART, "lgbm_fraud_model.pkl"))
scaler        = joblib.load(os.path.join(ART, "scaler.pkl"))
feature_cols  = joblib.load(os.path.join(ART, "feature_columns.pkl"))
num_features  = joblib.load(os.path.join(ART, "num_features.pkl"))

app = Flask(__name__)
CORS(app)

# ── Dataset statistics for synthetic generation (derived from Kaggle dataset) ─
CATEGORIES = [
    "gas_transport", "grocery_pos", "home", "shopping_pos", "kids_pets",
    "shopping_net", "personal_care", "entertainment", "food_dining",
    "health_fitness", "misc_pos", "misc_net", "grocery_net", "travel"
]
CATEGORY_WEIGHTS = [
    35089, 32732, 32516, 30329, 29704, 26379, 24406, 24222, 23038,
    22593, 20024, 16898, 11355, 10322
]
STATES = ["CA", "MO", "NE", "WY", "WA", "OR", "NM", "CO", "AZ", "UT", "ID", "HI", "AK"]
STATE_WEIGHTS = [80495, 54904, 34425, 27776, 27040, 26408, 23427, 19766, 15362, 15357, 8035, 3649, 2963]

# Lat/long bounding box from real dataset
LAT_RANGE  = (20.0271, 66.6933)
LON_RANGE  = (-165.6723, -89.6287)

# Merchant names pool (realistic examples)
MERCHANT_NAMES = [
    "Walmart", "Amazon", "Shell", "Chevron", "Target", "Costco",
    "McDonald's", "Starbucks", "Best Buy", "Home Depot", "CVS Pharmacy",
    "Walgreens", "Kroger", "Whole Foods", "Apple Store", "Netflix",
    "Uber", "Lyft", "Delta Airlines", "Marriott Hotels"
]

CITIES = ["Los Angeles", "Kansas City", "Omaha", "Cheyenne", "Seattle",
          "Portland", "Albuquerque", "Denver", "Phoenix", "Salt Lake City"]

JOBS = ["Engineer", "Teacher", "Nurse", "Accountant", "Manager",
        "Developer", "Artist", "Doctor", "Lawyer", "Scientist"]


def _weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices, weights):
        upto += w
        if r <= upto:
            return c
    return choices[-1]


def generate_synthetic_transaction(force_fraud: bool = False):
    """
    Generate a synthetic transaction based on real dataset statistics.
    Occasionally generates suspicious patterns that resemble fraud.
    """
    now = datetime.now() - timedelta(seconds=random.randint(0, 60))
    trans_dt = now.strftime("%Y-%m-%d %H:%M:%S")

    # Birth date → age between 18 and 85
    age_years = random.randint(18, 85)
    dob_dt = now - timedelta(days=age_years * 365 + random.randint(0, 364))
    dob = dob_dt.strftime("%Y-%m-%d")

    lat  = round(random.uniform(*LAT_RANGE), 4)
    lon  = round(random.uniform(*LON_RANGE), 4)

    state = _weighted_choice(STATES, STATE_WEIGHTS)
    category = _weighted_choice(CATEGORIES, CATEGORY_WEIGHTS)

    if force_fraud:
        # Fraud patterns: very high amount, large merchant distance, unusual hour
        amt = round(random.uniform(800, 5000), 2)
        merch_lat  = round(lat + random.uniform(5, 15) * random.choice([-1, 1]), 4)
        merch_long = round(lon + random.uniform(5, 20) * random.choice([-1, 1]), 4)
        # Odd hours: late night / early morning
        hour_bias = random.choice([1, 2, 3, 4, 23])
        trans_dt = now.replace(hour=hour_bias, minute=random.randint(0, 59)).strftime("%Y-%m-%d %H:%M:%S")
        city_pop = random.randint(46, 500)
    else:
        # Normal patterns: low-mid amount, close merchant
        amt = round(np.random.lognormal(mean=3.5, sigma=1.2), 2)
        amt = max(1.0, min(amt, 500.0))
        merch_lat  = round(lat + random.uniform(-0.5, 0.5), 6)
        merch_long = round(lon + random.uniform(-0.5, 0.5), 6)
        city_pop = int(np.random.lognormal(mean=7.5, sigma=2.0))
        city_pop = max(46, min(city_pop, 2383912))

    return {
        "trans_date_trans_time": trans_dt,
        "merchant": random.choice(MERCHANT_NAMES),
        "category": category,
        "amt": amt,
        "city": random.choice(CITIES),
        "state": state,
        "lat": lat,
        "long": lon,
        "city_pop": city_pop,
        "job": random.choice(JOBS),
        "dob": dob,
        "trans_num": str(uuid.uuid4()).replace("-", ""),
        "merch_lat": merch_lat,
        "merch_long": merch_long,
    }


def haversine_approx(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def engineer_features(raw: dict) -> pd.DataFrame:
    """Apply the same feature-engineering pipeline used during training."""
    r = raw.copy()

    # 1. Datetime features
    dt = pd.to_datetime(r["trans_date_trans_time"])
    r["hour"]        = dt.hour
    r["day_of_week"] = dt.dayofweek
    r["month"]       = dt.month

    # 2. Age
    dob = pd.to_datetime(r["dob"])
    r["age"] = (dt - dob).days // 365

    # 3. Distance
    r["distance_km"] = haversine_approx(
        r["lat"], r["long"], r["merch_lat"], r["merch_long"]
    )

    # 4. Log-amount
    r["log_amt"] = np.log1p(r["amt"])

    # 5. Drop raw columns not used by model
    for col in ["trans_date_trans_time", "dob", "merchant", "city", "trans_num", "job"]:
        r.pop(col, None)

    df = pd.DataFrame([r])

    # 6. One-hot encode
    df = pd.get_dummies(df, columns=["category", "state"], drop_first=False, dtype=int)

    # 7. Align to training columns (fill missing OHE columns with 0)
    df = df.reindex(columns=feature_cols, fill_value=0)

    # 8. Scale numeric features
    df[num_features] = scaler.transform(df[num_features])

    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "LightGBM Fraud Detector v1.0"})


@app.route("/generate", methods=["GET"])
def generate():
    """Generate a synthetic transaction. Pass ?fraud=1 to force fraudulent pattern."""
    force_fraud_flag = request.args.get("fraud", "0") == "1"
    # Randomly produce ~5% fraud rate unless forced
    is_forced = force_fraud_flag or (random.random() < 0.05)
    tx = generate_synthetic_transaction(force_fraud=is_forced)
    return jsonify(tx)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty request body"}), 400

    required = ["trans_date_trans_time", "amt", "lat", "long",
                "merch_lat", "merch_long", "category", "state", "dob", "city_pop"]
    missing  = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 422

    try:
        features = engineer_features(payload)
        prob     = float(model.predict_proba(features)[0, 1])
        pred     = int(prob >= 0.5)

        # Build engineered feature snapshot for educational display
        dt  = pd.to_datetime(payload["trans_date_trans_time"])
        dob = pd.to_datetime(payload["dob"])
        age = (dt - dob).days // 365
        distance_km = haversine_approx(
            payload["lat"], payload["long"], payload["merch_lat"], payload["merch_long"]
        )

        engineered = {
            "hour":        dt.hour,
            "day_of_week": dt.dayofweek,
            "month":       dt.month,
            "age":         int(age),
            "distance_km": round(distance_km, 2),
            "log_amt":     round(float(np.log1p(payload["amt"])), 4),
            "category":    payload.get("category", ""),
            "state":       payload.get("state", ""),
        }

        return jsonify({
            "is_fraud":          pred,
            "fraud_probability": round(prob, 6),
            "verdict":           "FRAUD" if pred == 1 else "Legitimate",
            "threshold":         0.5,
            "engineered_features": engineered,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Fraud Detection API on http://127.0.0.1:5050")
    app.run(debug=False, host="0.0.0.0", port=5050)
