"""
model_test.py — ML Model Tests (no API, no Flask)
==================================================
Tests cover three phases:
  Phase 1 — Artefact loading  (this file: complete)
  Phase 2 — Feature engineering logic
  Phase 3 — Model inference & quality

Run from the project root:
    python model_test.py
"""

import os
import unittest
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
ARTEFACT_DIR = os.path.join(BASE, "model_artefacts")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Artefact Loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtefacts(unittest.TestCase):
    """Verify that all saved model artefacts exist and load correctly."""

    def setUp(self):
        """Load all four artefacts once before each test method."""
        self.model        = joblib.load(os.path.join(ARTEFACT_DIR, "lgbm_fraud_model.pkl"))
        self.scaler       = joblib.load(os.path.join(ARTEFACT_DIR, "scaler.pkl"))
        self.feature_cols = joblib.load(os.path.join(ARTEFACT_DIR, "feature_columns.pkl"))
        self.num_features = joblib.load(os.path.join(ARTEFACT_DIR, "num_features.pkl"))

    # ── Test 1: all four .pkl files exist on disk ──────────────────────────────
    def test_artefact_files_exist(self):
        """All four artefact files must be present."""
        expected = [
            "lgbm_fraud_model.pkl",
            "scaler.pkl",
            "feature_columns.pkl",
            "num_features.pkl",
        ]
        for fname in expected:
            path = os.path.join(ARTEFACT_DIR, fname)
            self.assertTrue(os.path.isfile(path), f"Missing artefact: {fname}")

    # ── Test 2: feature_columns is a non-empty list of strings ─────────────────
    def test_feature_columns_type(self):
        """feature_columns must be a non-empty list of strings."""
        self.assertIsInstance(self.feature_cols, list)
        self.assertGreater(len(self.feature_cols), 0, "feature_columns is empty")
        for col in self.feature_cols:
            self.assertIsInstance(col, str, f"Non-string entry in feature_cols: {col!r}")

    # ── Test 3: num_features is a subset of feature_cols ──────────────────────
    def test_num_features_subset_of_feature_cols(self):
        """Every numeric feature must appear in the full feature column list."""
        self.assertIsInstance(self.num_features, list)
        self.assertGreater(len(self.num_features), 0, "num_features is empty")
        feature_set = set(self.feature_cols)
        for col in self.num_features:
            self.assertIn(
                col, feature_set,
                f"num_feature '{col}' not found in feature_columns — pipeline mismatch"
            )

    # ── Test 4: scaler has a transform method ─────────────────────────────────
    def test_scaler_has_transform(self):
        """Scaler must be a fitted sklearn transformer with a transform() method."""
        self.assertTrue(
            hasattr(self.scaler, "transform"),
            "Loaded scaler does not have a transform() method"
        )

    # ── Test 5: model has predict and predict_proba methods ───────────────────
    def test_model_has_predict_methods(self):
        """Model must expose predict() and predict_proba()."""
        self.assertTrue(
            hasattr(self.model, "predict"),
            "Model does not have a predict() method"
        )
        self.assertTrue(
            hasattr(self.model, "predict_proba"),
            "Model does not have a predict_proba() method"
        )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
