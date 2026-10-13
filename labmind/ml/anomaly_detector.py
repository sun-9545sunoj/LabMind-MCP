"""
Lab Anomaly Detector — Isolation Forest multi-parameter anomaly detection.

Identifies subtle multi-parameter anomalies in lab panels that may not be
apparent when evaluating individual tests against reference ranges.

Example:
    >>> detector = LabAnomalyDetector()
    >>> detector.train(historical_lab_df)
    >>> result = detector.detect({"wbc": 14.2, "hemoglobin": 11.5, ...})
    >>> print(result.is_anomaly, result.flagged_parameters)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from labmind.config import (
    ANOMALY_CONTAMINATION,
    ANOMALY_FEATURES,
    ANOMALY_MODEL_PATH,
    ANOMALY_RANDOM_STATE,
)

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of an anomaly detection run on a lab panel."""

    is_anomaly: bool
    anomaly_score: float
    confidence: float
    flagged_parameters: list[str] = field(default_factory=list)
    z_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class LabAnomalyDetector:
    """
    Multi-parameter lab anomaly detector using Isolation Forest.

    Detects anomalous lab panel combinations that deviate from historical
    patterns, even when individual values may appear within normal ranges.
    """

    FEATURES = ANOMALY_FEATURES

    def __init__(self) -> None:
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._training_mean: Optional[np.ndarray] = None
        self._training_std: Optional[np.ndarray] = None
        self._is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, lab_data: pd.DataFrame) -> dict[str, float]:
        """
        Train the Isolation Forest on historical lab panel data.

        Args:
            lab_data: DataFrame with columns matching ANOMALY_FEATURES.

        Returns:
            Training metrics dict with anomaly_rate, n_samples, n_features.
        """
        missing = [f for f in self.FEATURES if f not in lab_data.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = lab_data[self.FEATURES].values.astype(np.float64)

        # Handle NaN values
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_means[i]

        # Store statistics for z-score analysis
        self._training_mean = np.mean(X, axis=0)
        self._training_std = np.std(X, axis=0)
        self._training_std[self._training_std == 0] = 1.0  # Avoid division by zero

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train Isolation Forest
        self._model = IsolationForest(
            contamination=ANOMALY_CONTAMINATION,
            random_state=ANOMALY_RANDOM_STATE,
            n_estimators=200,
            max_samples="auto",
        )
        self._model.fit(X_scaled)
        self._is_fitted = True

        # Compute training metrics
        predictions = self._model.predict(X_scaled)
        anomaly_rate = float(np.mean(predictions == -1))

        metrics = {
            "anomaly_rate": anomaly_rate,
            "n_samples": len(X),
            "n_features": len(self.FEATURES),
            "contamination": ANOMALY_CONTAMINATION,
        }
        logger.info("Anomaly detector trained: %s", metrics)
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def detect(self, lab_values: dict[str, float]) -> AnomalyResult:
        """
        Run anomaly detection on a single lab panel.

        Args:
            lab_values: Dict mapping feature names to numeric values.

        Returns:
            AnomalyResult with anomaly flag, score, and flagged parameters.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call train() or load_model() first."
            )

        # Build feature vector (use training mean for missing values)
        features = []
        for i, feat in enumerate(self.FEATURES):
            val = lab_values.get(feat)
            if val is None:
                val = float(self._training_mean[i])  # type: ignore[index]
                logger.warning("Missing feature '%s', using training mean.", feat)
            features.append(float(val))

        X = np.array([features], dtype=np.float64)
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]

        # Predict
        score = float(self._model.decision_function(X_scaled)[0])  # type: ignore[union-attr]
        prediction = int(self._model.predict(X_scaled)[0])  # type: ignore[union-attr]
        is_anomaly = prediction == -1

        # Confidence: normalize score to 0–1 range (lower score → higher confidence)
        # Typical decision function range is roughly [-0.5, 0.5]
        confidence = min(1.0, max(0.0, 0.5 - score))

        # Z-score analysis: identify which parameters are driving the anomaly
        z_scores: dict[str, float] = {}
        flagged: list[str] = []

        for i, feat in enumerate(self.FEATURES):
            z = (features[i] - self._training_mean[i]) / self._training_std[i]  # type: ignore[index]
            z_scores[feat] = round(float(z), 3)
            if abs(z) > 2.0:
                flagged.append(feat)

        # Sort flagged parameters by absolute z-score (most deviant first)
        flagged.sort(key=lambda f: abs(z_scores[f]), reverse=True)

        # Generate explanation
        if is_anomaly:
            if flagged:
                details = ", ".join(
                    f"{f} (z={z_scores[f]:+.2f})" for f in flagged[:4]
                )
                explanation = (
                    f"ANOMALY DETECTED: Multi-parameter deviation identified. "
                    f"Key outlier parameters: {details}. "
                    f"Anomaly score: {score:.4f}."
                )
            else:
                explanation = (
                    f"ANOMALY DETECTED: Unusual combination pattern identified "
                    f"(no single parameter exceeds 2σ, but multivariate profile "
                    f"is atypical). Anomaly score: {score:.4f}."
                )
        else:
            explanation = (
                f"Lab panel within expected multivariate range. "
                f"Anomaly score: {score:.4f}."
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            confidence=round(confidence, 4),
            flagged_parameters=flagged,
            z_scores=z_scores,
            explanation=explanation,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save_model(self, path: Path | None = None) -> None:
        """Save the trained model, scaler, and statistics to disk."""
        if not self._is_fitted:
            raise RuntimeError("No fitted model to save.")

        save_path = path or ANOMALY_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": self._model,
            "scaler": self._scaler,
            "training_mean": self._training_mean,
            "training_std": self._training_std,
        }
        joblib.dump(bundle, save_path)
        logger.info("Anomaly detector saved to %s", save_path)

    def load_model(self, path: Path | None = None) -> None:
        """Load a previously trained model from disk."""
        load_path = path or ANOMALY_MODEL_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"No model file at {load_path}")

        bundle = joblib.load(load_path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._training_mean = bundle["training_mean"]
        self._training_std = bundle["training_std"]
        self._is_fitted = True
        logger.info("Anomaly detector loaded from %s", load_path)
