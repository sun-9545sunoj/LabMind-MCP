"""
Sepsis Risk Predictor — XGBoost classifier with SHAP explainability.

Predicts the probability of sepsis from lab values and vitals, returning
risk levels, SHAP feature importances, and clinical recommendations.

Example:
    >>> predictor = SepsisRiskPredictor()
    >>> predictor.train(X_train, y_train)
    >>> result = predictor.predict({"wbc": 15.0, "lactate": 4.1, ...})
    >>> print(result.risk_level, result.top_risk_factors)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from labmind.config import (
    SEPSIS_ALL_FEATURES,
    SEPSIS_LEARNING_RATE,
    SEPSIS_MAX_DEPTH,
    SEPSIS_MODEL_PATH,
    SEPSIS_N_ESTIMATORS,
    SEPSIS_RISK_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class SepsisRiskResult:
    """Result of a sepsis risk prediction."""

    probability: float
    risk_level: str
    shap_values: dict[str, float] = field(default_factory=dict)
    top_risk_factors: list[dict[str, str]] = field(default_factory=list)
    recommendation: str = ""


def generate_sepsis_training_data(
    n_samples: int = 2000,
    sepsis_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate realistic synthetic training data for the sepsis classifier.

    Sepsis-positive samples have elevated WBC, Lactate, Creatinine, HR, RR,
    Temperature, and depressed Platelets & Systolic BP.

    Args:
        n_samples: Total number of training samples.
        sepsis_ratio: Fraction of positive (sepsis) cases.
        seed: Random seed for reproducibility.

    Returns:
        (X, y) where X is a DataFrame and y is a binary Series.
    """
    rng = np.random.RandomState(seed)
    n_pos = int(n_samples * sepsis_ratio)
    n_neg = n_samples - n_pos

    # Normal patients
    neg = pd.DataFrame({
        "wbc":              rng.normal(7.5, 1.5, n_neg),
        "hemoglobin":       rng.normal(14.5, 1.5, n_neg),
        "platelets":        rng.normal(250.0, 50.0, n_neg),
        "creatinine":       rng.normal(1.0, 0.15, n_neg),
        "lactate":          rng.normal(1.2, 0.4, n_neg),
        "bun":              rng.normal(14.0, 3.0, n_neg),
        "glucose":          rng.normal(90.0, 10.0, n_neg),
        "potassium":        rng.normal(4.2, 0.3, n_neg),
        "heart_rate":       rng.normal(75.0, 8.0, n_neg),
        "systolic_bp":      rng.normal(120.0, 10.0, n_neg),
        "temperature":      rng.normal(36.8, 0.3, n_neg),
        "respiratory_rate": rng.normal(16.0, 2.0, n_neg),
    })

    # Sepsis patients
    pos = pd.DataFrame({
        "wbc":              rng.normal(16.0, 4.0, n_pos),
        "hemoglobin":       rng.normal(11.0, 2.0, n_pos),
        "platelets":        rng.normal(120.0, 40.0, n_pos),
        "creatinine":       rng.normal(2.5, 0.8, n_pos),
        "lactate":          rng.normal(4.5, 1.5, n_pos),
        "bun":              rng.normal(35.0, 10.0, n_pos),
        "glucose":          rng.normal(140.0, 30.0, n_pos),
        "potassium":        rng.normal(5.2, 0.6, n_pos),
        "heart_rate":       rng.normal(110.0, 15.0, n_pos),
        "systolic_bp":      rng.normal(85.0, 12.0, n_pos),
        "temperature":      rng.normal(38.8, 0.6, n_pos),
        "respiratory_rate": rng.normal(24.0, 4.0, n_pos),
    })

    X = pd.concat([neg, pos], ignore_index=True)
    y = pd.Series([0] * n_neg + [1] * n_pos, name="sepsis")

    # Shuffle
    idx = rng.permutation(len(X))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


class SepsisRiskPredictor:
    """
    XGBoost-based sepsis risk classifier with SHAP explainability.

    Combines lab values and vital signs to predict sepsis probability,
    providing interpretable risk assessments for clinical decision support.
    """

    FEATURES = SEPSIS_ALL_FEATURES

    def __init__(self) -> None:
        self._model: Optional[xgb.XGBClassifier] = None
        self._is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """
        Train the XGBoost sepsis classifier.

        Args:
            X: Feature DataFrame with columns matching SEPSIS_ALL_FEATURES.
            y: Binary target Series (0 = no sepsis, 1 = sepsis).

        Returns:
            Training metrics dict.
        """
        missing = [f for f in self.FEATURES if f not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Handle class imbalance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        self._model = xgb.XGBClassifier(
            n_estimators=SEPSIS_N_ESTIMATORS,
            max_depth=SEPSIS_MAX_DEPTH,
            learning_rate=SEPSIS_LEARNING_RATE,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )

        X_train = X[self.FEATURES]
        self._model.fit(X_train, y)
        self._is_fitted = True

        # Training metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        y_pred = self._model.predict(X_train)
        metrics = {
            "accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y, y_pred, zero_division=0)), 4),
            "n_samples": len(y),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "scale_pos_weight": round(scale_pos_weight, 2),
        }
        logger.info("Sepsis predictor trained: %s", metrics)
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, lab_values: dict[str, float]) -> SepsisRiskResult:
        """
        Predict sepsis risk from lab values and vitals.

        Args:
            lab_values: Dict mapping feature names to numeric values.
                Lab features: wbc, hemoglobin, platelets, creatinine,
                    lactate, bun, glucose, potassium
                Vitals: heart_rate, systolic_bp, temperature, respiratory_rate

        Returns:
            SepsisRiskResult with probability, risk level, SHAP values,
            top risk factors, and clinical recommendation.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call train() or load_model() first."
            )

        # Build feature vector with defaults for missing vitals
        defaults = {
            "wbc": 7.5, "hemoglobin": 14.5, "platelets": 250.0,
            "creatinine": 1.0, "lactate": 1.2, "bun": 14.0,
            "glucose": 90.0, "potassium": 4.2, "heart_rate": 75.0,
            "systolic_bp": 120.0, "temperature": 36.8, "respiratory_rate": 16.0,
        }

        features = []
        for feat in self.FEATURES:
            val = lab_values.get(feat, defaults.get(feat, 0.0))
            features.append(float(val))

        X = np.array([features], dtype=np.float64)
        X_df = pd.DataFrame(X, columns=self.FEATURES)

        # Probability prediction
        prob = float(self._model.predict_proba(X_df)[0][1])

        # Risk level
        risk_level = "CRITICAL"
        for level, threshold in sorted(
            SEPSIS_RISK_THRESHOLDS.items(),
            key=lambda x: x[1],
        ):
            if prob < threshold:
                risk_level = level
                break

        # SHAP values for explainability
        shap_vals: dict[str, float] = {}
        top_factors: list[dict[str, str]] = []

        try:
            import shap

            explainer = shap.TreeExplainer(self._model)
            sv = explainer.shap_values(X_df)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            shap_array = sv[0] if sv.ndim > 1 else sv

            for i, feat in enumerate(self.FEATURES):
                shap_vals[feat] = round(float(shap_array[i]), 4)

            # Top 3 risk factors by absolute SHAP value
            sorted_feats = sorted(
                shap_vals.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for feat_name, shap_val in sorted_feats[:3]:
                direction = "↑ increases risk" if shap_val > 0 else "↓ decreases risk"
                feat_val = lab_values.get(feat_name, defaults.get(feat_name, 0.0))
                top_factors.append({
                    "feature": feat_name,
                    "value": str(round(feat_val, 2)),
                    "shap_contribution": str(round(shap_val, 4)),
                    "direction": direction,
                })

        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)
            # Fallback: use feature importances
            importances = self._model.feature_importances_
            for i, feat in enumerate(self.FEATURES):
                shap_vals[feat] = round(float(importances[i]), 4)

        # Clinical recommendation
        recommendation = self._get_recommendation(risk_level, prob, top_factors)

        return SepsisRiskResult(
            probability=round(prob, 4),
            risk_level=risk_level,
            shap_values=shap_vals,
            top_risk_factors=top_factors,
            recommendation=recommendation,
        )

    @staticmethod
    def _get_recommendation(
        risk_level: str,
        prob: float,
        top_factors: list[dict[str, str]],
    ) -> str:
        """Generate a clinical recommendation based on risk assessment."""
        if risk_level == "CRITICAL":
            return (
                "⚠️ CRITICAL SEPSIS RISK (probability: {:.1%}). "
                "Recommend immediate clinical review, blood cultures, "
                "and initiation of empiric broad-spectrum antibiotics per "
                "Surviving Sepsis Campaign guidelines. Consider ICU consult."
            ).format(prob)
        elif risk_level == "HIGH":
            return (
                "🔴 HIGH SEPSIS RISK (probability: {:.1%}). "
                "Recommend urgent assessment. Consider blood cultures, "
                "lactate monitoring q2h, and antibiotic initiation within 1 hour "
                "if clinical suspicion is confirmed."
            ).format(prob)
        elif risk_level == "MODERATE":
            return (
                "🟡 MODERATE SEPSIS RISK (probability: {:.1%}). "
                "Recommend close monitoring with repeat labs in 4-6 hours. "
                "Assess for clinical signs of infection. "
                "Consider sepsis screening protocol."
            ).format(prob)
        else:
            return (
                "🟢 LOW SEPSIS RISK (probability: {:.1%}). "
                "Current lab values and vitals within expected parameters. "
                "Continue routine monitoring per standard of care."
            ).format(prob)

    # ── Persistence ───────────────────────────────────────────────────────

    def save_model(self, path: Path | None = None) -> None:
        """Save the trained model to disk."""
        if not self._is_fitted:
            raise RuntimeError("No fitted model to save.")

        save_path = path or SEPSIS_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, save_path)
        logger.info("Sepsis predictor saved to %s", save_path)

    def load_model(self, path: Path | None = None) -> None:
        """Load a previously trained model from disk."""
        load_path = path or SEPSIS_MODEL_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"No model file at {load_path}")

        self._model = joblib.load(load_path)
        self._is_fitted = True
        logger.info("Sepsis predictor loaded from %s", load_path)
