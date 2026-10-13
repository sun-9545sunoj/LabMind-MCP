"""
Model training script for LabMind-MCP.

Loads data from the SQLite database, trains the anomaly detector and
sepsis risk predictor, saves models, and reports training metrics.

Usage:
    python -m labmind.ml.train_models
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from labmind.config import ANOMALY_FEATURES, DB_PATH, MODELS_DIR
from labmind.db.database import LabDatabase
from labmind.ml.anomaly_detector import LabAnomalyDetector
from labmind.ml.sepsis_predictor import (
    SepsisRiskPredictor,
    generate_sepsis_training_data,
)


def train_anomaly_detector(db: LabDatabase) -> dict[str, float]:
    """Train the Isolation Forest anomaly detector from historical lab data."""
    print("\n" + "=" * 60)
    print("🔬 Training Lab Anomaly Detector (Isolation Forest)")
    print("=" * 60)

    # Collect latest lab panels from all patients
    patients = db.get_all_patients()
    rows: list[dict[str, float]] = []

    for patient in patients:
        panel = db.get_latest_lab_panel(patient.patient_id)
        if not panel:
            continue

        # Map test names to feature names (lowercase)
        feature_map: dict[str, float] = {}
        for test_name, result in panel.items():
            key = test_name.lower()
            if key in ANOMALY_FEATURES:
                feature_map[key] = result.value

        # Only include if we have enough features
        if len(feature_map) >= len(ANOMALY_FEATURES) * 0.7:
            rows.append(feature_map)

    if not rows:
        print("❌ No suitable training data found. Seed the database first.")
        return {}

    df = pd.DataFrame(rows)

    # Fill missing columns with column means
    for feat in ANOMALY_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
    df = df[ANOMALY_FEATURES].fillna(df.mean())

    print(f"   Training samples: {len(df)}")
    print(f"   Features: {list(df.columns)}")

    detector = LabAnomalyDetector()
    metrics = detector.train(df)
    detector.save_model()

    print(f"   Anomaly rate: {metrics['anomaly_rate']:.2%}")
    print(f"   ✅ Model saved to {MODELS_DIR / 'anomaly_detector.joblib'}")

    return metrics


def train_sepsis_predictor() -> dict[str, float]:
    """Train the XGBoost sepsis risk predictor with synthetic data."""
    print("\n" + "=" * 60)
    print("🦠 Training Sepsis Risk Predictor (XGBoost)")
    print("=" * 60)

    X, y = generate_sepsis_training_data(n_samples=2000, sepsis_ratio=0.25)

    print(f"   Training samples: {len(X)}")
    print(f"   Positive (sepsis): {int(y.sum())} ({y.mean():.1%})")
    print(f"   Negative (healthy): {int((1 - y).sum())}")
    print(f"   Features: {list(X.columns)}")

    predictor = SepsisRiskPredictor()
    metrics = predictor.train(X, y)
    predictor.save_model()

    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   ✅ Model saved to {MODELS_DIR / 'sepsis_predictor.joblib'}")

    return metrics


def main() -> None:
    """Run the full training pipeline."""
    print("🧬 LabMind-MCP Model Training Pipeline")
    print("=" * 60)

    # Check database exists
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        print("   Run: python -m labmind.data.seed_database")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    db = LabDatabase()

    try:
        anomaly_metrics = train_anomaly_detector(db)
        sepsis_metrics = train_sepsis_predictor()

        print("\n" + "=" * 60)
        print("✅ All models trained and saved successfully!")
        print("=" * 60)
        print(f"\n   Models directory: {MODELS_DIR}")
        print(f"   Ready to start the MCP server: python -m labmind.server")

    finally:
        db.close()


if __name__ == "__main__":
    main()
