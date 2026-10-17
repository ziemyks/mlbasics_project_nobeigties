"""
fold_scaler_into_model.py — remove the scaling step from the inference pipeline
================================================================================
The LightGBM model was trained on StandardScaler output, so every split threshold
on a numeric feature is expressed in z-score units. Trees only compare a value with
a threshold, and z = (x − μ) / σ is a monotonic (σ > 0) linear transform, so

    z <= t   ⇔   x <= t·σ + μ

Rewriting each numeric threshold as t·σ + μ gives a model that takes raw (unscaled)
features and makes exactly the same decisions. One-hot columns are left as they are.

Run:
    python fold_scaler_into_model.py
Writes:
    model_artefacts/lgbm_fraud_model_unscaled.txt   (LightGBM text format)
"""

import os
import joblib
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "model_artefacts")

model = joblib.load(os.path.join(ART, "lgbm_fraud_model.pkl"))
scaler = joblib.load(os.path.join(ART, "scaler.pkl"))
feature_cols = joblib.load(os.path.join(ART, "feature_columns.pkl"))
num_features = joblib.load(os.path.join(ART, "num_features.pkl"))

# feature index (model column order) → (mean, std) of the scaler
stats = {feature_cols.index(f): (scaler.mean_[i], scaler.scale_[i])
         for i, f in enumerate(num_features)}


def unscale(idx, value):
    mu, sigma = stats[idx]
    return float(value * sigma + mu)


out, split_features = [], None
for line in model.booster_.model_to_string().splitlines():
    if line.startswith("tree_sizes="):
        continue  # byte offsets of each tree change; LightGBM parses sequentially without them
    if line.startswith("decision_type="):
        # every split must be numerical with no zero-as-missing handling
        assert set(line.split("=")[1].split()) <= {"0", "2"}, line
    if line.startswith("split_feature="):
        split_features = [int(v) for v in line.split("=")[1].split()]
    elif line.startswith("threshold=") and split_features is not None:
        thresholds = [float(v) for v in line.split("=")[1].split()]
        line = "threshold=" + " ".join(
            repr(unscale(f, t)) if f in stats else repr(t)
            for f, t in zip(split_features, thresholds))
        split_features = None
    elif line.startswith("feature_infos="):
        infos = line.split("=", 1)[1].split()
        for idx in stats:
            lo, hi = (float(v) for v in infos[idx].strip("[]").split(":"))
            infos[idx] = f"[{repr(unscale(idx, lo))}:{repr(unscale(idx, hi))}]"
        line = "feature_infos=" + " ".join(infos)
    out.append(line)

booster = lgb.Booster(model_str="\n".join(out) + "\n")
path = os.path.join(ART, "lgbm_fraud_model_unscaled.txt")
booster.save_model(path)
print(f"Saved {path}")
