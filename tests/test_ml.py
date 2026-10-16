"""ML model behavior on clearly separated inputs."""

from datetime import datetime, timedelta

import pytest

from labmind.ml.anomaly_detector import LabAnomalyDetector
from labmind.ml.trend_analyzer import LabTrendAnalyzer

NORMAL_PANEL = {
    "wbc": 7.5, "hemoglobin": 14.5, "platelets": 250.0, "creatinine": 1.0,
    "lactate": 1.2, "bun": 14.0, "glucose": 90.0, "potassium": 4.2,
}
NORMAL_VITALS = {
    "heart_rate": 75.0, "systolic_bp": 120.0,
    "temperature": 36.8, "respiratory_rate": 16.0,
}
SEPTIC = {
    **NORMAL_PANEL, "wbc": 18.0, "platelets": 90.0, "creatinine": 2.5,
    "lactate": 5.0, "heart_rate": 120.0, "systolic_bp": 80.0,
    "temperature": 39.2, "respiratory_rate": 28.0,
}


def test_anomaly_detector_requires_fitting():
    with pytest.raises(RuntimeError):
        LabAnomalyDetector().detect(NORMAL_PANEL)


def test_anomaly_detector_flags_extreme_panel(anomaly_detector):
    extreme = {**NORMAL_PANEL, "wbc": 28.0, "lactate": 7.5, "platelets": 40.0}
    result = anomaly_detector.detect(extreme)
    assert result.is_anomaly
    assert {"wbc", "lactate"} & set(result.flagged_parameters)


def test_anomaly_detector_passes_typical_panel(anomaly_detector):
    assert not anomaly_detector.detect(NORMAL_PANEL).is_anomaly


def test_anomaly_training_uses_all_historical_panels(db):
    from labmind.ml.train_models import build_anomaly_training_frame

    frame = build_anomaly_training_frame(db)
    # Every patient has at least 3 collection times
    assert len(frame) >= 3 * len(db.get_all_patients())
    assert not frame.isna().any().any()


def test_sepsis_risk_orders_cases(sepsis_predictor):
    healthy = sepsis_predictor.predict({**NORMAL_PANEL, **NORMAL_VITALS})
    septic = sepsis_predictor.predict(SEPTIC)
    assert healthy.risk_level == "LOW"
    assert septic.risk_level in {"HIGH", "CRITICAL"}
    assert septic.probability > healthy.probability
    assert septic.top_risk_factors


def _series(values, step_days=2):
    start = datetime(2026, 1, 1)
    return values, [start + timedelta(days=i * step_days) for i in range(len(values))]


def test_trend_detects_rise():
    values, times = _series([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    trend = LabTrendAnalyzer().analyze_trend(values, times, "Creatinine")
    assert "INCREASING" in trend.trend_direction
    assert trend.slope > 0
    assert trend.r_squared > 0.99


def test_trend_stable():
    values, times = _series([1.0, 1.01, 0.99, 1.0, 1.01, 0.99])
    trend = LabTrendAnalyzer().analyze_trend(values, times, "Creatinine")
    assert trend.trend_direction == "STABLE"


def test_trend_insufficient_data():
    values, times = _series([1.0])
    trend = LabTrendAnalyzer().analyze_trend(values, times, "Creatinine")
    assert trend.trend_direction == "INSUFFICIENT_DATA"
