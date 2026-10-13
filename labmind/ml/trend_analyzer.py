"""
Lab Trend Analyzer — temporal trend analysis with linear regression.

Computes slope, direction, projected values, and clinical significance
for longitudinal lab test trajectories.

Example:
    >>> analyzer = LabTrendAnalyzer()
    >>> result = analyzer.analyze_trend(values, timestamps, "Creatinine")
    >>> print(result.trend_direction, result.projected_values)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrendResult:
    """Result of a trend analysis on a time-series of lab values."""

    trend_direction: str  # STABLE, INCREASING, DECREASING, RAPIDLY_INCREASING, RAPIDLY_DECREASING
    slope: float  # Change per day
    slope_per_week: float
    r_squared: float
    n_datapoints: int
    date_range_days: int
    projected_values: dict[str, float] = field(default_factory=dict)
    clinical_significance: str = ""
    explanation: str = ""


class LabTrendAnalyzer:
    """
    Temporal trend analysis for laboratory test results.

    Uses linear regression to compute slope, detect acceleration,
    classify trend direction, and project future values with
    clinical significance assessment.
    """

    # Slope thresholds per day for classification (as fraction of typical range)
    RAPID_THRESHOLD = 0.02    # 2% of typical range per day
    CHANGE_THRESHOLD = 0.005  # 0.5% of typical range per day

    # Typical physiological ranges for slope normalization
    TYPICAL_RANGES: dict[str, tuple[float, float]] = {
        "WBC":           (4.5, 11.0),
        "Hemoglobin":    (12.0, 17.5),
        "Hematocrit":    (36.0, 52.0),
        "Platelets":     (150.0, 450.0),
        "Creatinine":    (0.74, 1.35),
        "BUN":           (7.0, 20.0),
        "Glucose":       (70.0, 100.0),
        "Sodium":        (136.0, 145.0),
        "Potassium":     (3.5, 5.0),
        "Lactate":       (0.5, 2.2),
        "Procalcitonin": (0.0, 0.1),
        "CRP":           (0.0, 5.0),
        "Calcium":       (8.5, 10.5),
        "Chloride":      (98.0, 106.0),
        "CO2":           (23.0, 29.0),
        "eGFR":          (60.0, 120.0),
    }

    # Tests where DECREASE is clinically concerning
    CONCERN_ON_DECREASE = {"eGFR", "Hemoglobin", "Platelets", "Hematocrit"}
    # Tests where INCREASE is clinically concerning
    CONCERN_ON_INCREASE = {
        "Creatinine", "BUN", "Lactate", "WBC", "Procalcitonin",
        "CRP", "Glucose", "Potassium",
    }

    def analyze_trend(
        self,
        values: list[float],
        timestamps: list[datetime],
        test_name: str,
    ) -> TrendResult:
        """
        Analyze the temporal trend of a lab test series.

        Args:
            values: Chronologically ordered lab values.
            timestamps: Corresponding collection timestamps.
            test_name: Name of the lab test (for clinical context).

        Returns:
            TrendResult with direction, slope, projections, and significance.
        """
        if len(values) < 2:
            return TrendResult(
                trend_direction="INSUFFICIENT_DATA",
                slope=0.0,
                slope_per_week=0.0,
                r_squared=0.0,
                n_datapoints=len(values),
                date_range_days=0,
                explanation=f"Need at least 2 data points for trend analysis "
                            f"(received {len(values)}).",
            )

        if len(values) != len(timestamps):
            raise ValueError(
                f"values ({len(values)}) and timestamps ({len(timestamps)}) "
                f"must have equal length."
            )

        # Convert timestamps to numeric (days since first measurement)
        t0 = min(timestamps)
        t_days = np.array(
            [(t - t0).total_seconds() / 86400.0 for t in timestamps],
            dtype=np.float64,
        )
        y = np.array(values, dtype=np.float64)

        date_range_days = int(round(t_days[-1] - t_days[0]))

        # ── Linear regression ─────────────────────────────────────────────
        # y = slope * t + intercept
        n = len(y)
        t_mean = np.mean(t_days)
        y_mean = np.mean(y)

        numerator = np.sum((t_days - t_mean) * (y - y_mean))
        denominator = np.sum((t_days - t_mean) ** 2)

        if denominator < 1e-10:
            slope = 0.0
            intercept = y_mean
            r_squared = 0.0
        else:
            slope = float(numerator / denominator)
            intercept = float(y_mean - slope * t_mean)

            # R² calculation
            y_pred = slope * t_days + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        slope_per_week = slope * 7.0

        # ── Classify trend direction ──────────────────────────────────────
        typical_range = self.TYPICAL_RANGES.get(test_name, (0.0, 1.0))
        range_span = typical_range[1] - typical_range[0]
        if range_span <= 0:
            range_span = 1.0

        normalized_slope = abs(slope) / range_span

        if normalized_slope < self.CHANGE_THRESHOLD:
            trend_direction = "STABLE"
        elif slope > 0:
            if normalized_slope > self.RAPID_THRESHOLD:
                trend_direction = "RAPIDLY_INCREASING"
            else:
                trend_direction = "INCREASING"
        else:
            if normalized_slope > self.RAPID_THRESHOLD:
                trend_direction = "RAPIDLY_DECREASING"
            else:
                trend_direction = "DECREASING"

        # ── Projected values ──────────────────────────────────────────────
        last_t = t_days[-1]
        projections = {}
        for label, delta_days in [("7_days", 7), ("14_days", 14), ("30_days", 30)]:
            proj_val = slope * (last_t + delta_days) + intercept
            projections[label] = round(proj_val, 2)

        # ── Clinical significance ─────────────────────────────────────────
        significance, explanation = self._assess_significance(
            test_name, trend_direction, slope, slope_per_week,
            r_squared, values, projections, date_range_days, n,
        )

        return TrendResult(
            trend_direction=trend_direction,
            slope=round(slope, 4),
            slope_per_week=round(slope_per_week, 4),
            r_squared=round(r_squared, 4),
            n_datapoints=n,
            date_range_days=date_range_days,
            projected_values=projections,
            clinical_significance=significance,
            explanation=explanation,
        )

    def _assess_significance(
        self,
        test_name: str,
        direction: str,
        slope: float,
        slope_per_week: float,
        r_squared: float,
        values: list[float],
        projections: dict[str, float],
        date_range_days: int,
        n_points: int,
    ) -> tuple[str, str]:
        """Assess the clinical significance of the observed trend."""
        latest = values[-1]
        earliest = values[0]

        is_concerning = False
        if direction in ("INCREASING", "RAPIDLY_INCREASING"):
            is_concerning = test_name in self.CONCERN_ON_INCREASE
        elif direction in ("DECREASING", "RAPIDLY_DECREASING"):
            is_concerning = test_name in self.CONCERN_ON_DECREASE

        is_rapid = "RAPIDLY" in direction

        # Build significance label
        if direction == "STABLE":
            significance = "NOT_SIGNIFICANT"
        elif is_rapid and is_concerning:
            significance = "CRITICAL"
        elif is_concerning:
            significance = "CLINICALLY_SIGNIFICANT"
        elif is_rapid:
            significance = "MONITOR_CLOSELY"
        else:
            significance = "MILD_TREND"

        # Build explanation
        parts = [
            f"{test_name} trend over {date_range_days} days "
            f"({n_points} measurements): {direction}.",
            f"Change rate: {slope_per_week:+.3f}/week "
            f"(from {earliest:.2f} to {latest:.2f}).",
            f"R² = {r_squared:.3f} "
            f"({'strong fit' if r_squared > 0.7 else 'moderate fit' if r_squared > 0.4 else 'weak fit'}).",
        ]

        proj_7d = projections.get("7_days")
        proj_30d = projections.get("30_days")
        if proj_7d is not None and proj_30d is not None:
            parts.append(
                f"Projected: {proj_7d:.2f} in 7 days, {proj_30d:.2f} in 30 days."
            )

        if significance == "CRITICAL":
            parts.append(
                f"⚠️ CRITICAL: {test_name} is {direction.lower().replace('_', ' ')} — "
                f"warrants immediate clinical attention."
            )
        elif significance == "CLINICALLY_SIGNIFICANT":
            parts.append(
                f"🔴 Clinically significant {direction.lower().replace('_', ' ')} trend. "
                f"Recommend close monitoring and repeat testing."
            )
        elif significance == "MONITOR_CLOSELY":
            parts.append(
                f"🟡 Rapid change detected. Monitor closely and correlate clinically."
            )
        elif significance == "MILD_TREND":
            parts.append(
                f"📊 Mild trend observed. Continue routine monitoring."
            )
        else:
            parts.append(f"🟢 Values are stable. No intervention needed.")

        return significance, " ".join(parts)
