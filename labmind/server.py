"""
LabMind-MCP Server — Diagnostic Test Interpreter & Anomaly Detection.

Exposes laboratory diagnostic tools, reference data resources, and clinical
prompts via the Model Context Protocol (MCP) for integration with AI agents.

Usage:
    python -m labmind.server

Tools:
    - fetch_patient_lab_history: Retrieve lab results with abnormality flags
    - run_lab_anomaly_detector: Isolation Forest multi-parameter anomaly detection
    - predict_sepsis_risk: XGBoost sepsis risk prediction with SHAP explainability
    - analyze_lab_trend: Temporal trend analysis with projections
    - search_patients: Search patients by name or MRN
    - get_critical_alerts: Prioritized critical value alerts for a patient

Resources:
    - lis://reference-ranges: Standard reference ranges for all lab tests
    - lis://lab-codes: LOINC code directory with descriptions
    - lis://patient-roster: Patient discovery list

Prompts:
    - clinical_review: Comprehensive patient review workflow
    - shift_handoff: Multi-patient shift handoff report
"""

import logging
from datetime import datetime
from typing import Optional

try:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2.0
except ImportError:
    from mcp.server.fastmcp import FastMCP  # mcp 1.x

from labmind.config import (
    ANOMALY_FEATURES,
    LOINC_CODES,
    SERVER_NAME,
    SERVER_VERSION,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Initialize MCP Server ────────────────────────────────────────────────────

mcp = FastMCP(SERVER_NAME)

# Normal vitals used when predict_sepsis_risk is called without measurements
DEFAULT_VITALS = {
    "heart_rate": 80.0,
    "systolic_bp": 120.0,
    "temperature": 37.0,
    "respiratory_rate": 16.0,
}

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_db = None
_anomaly_detector = None
_sepsis_predictor = None
_trend_analyzer = None


def _get_db():
    """Lazy-load the database connection."""
    global _db
    if _db is None:
        from labmind.db.database import LabDatabase
        _db = LabDatabase()
    return _db


def _get_anomaly_detector():
    """Lazy-load the anomaly detector model."""
    global _anomaly_detector
    if _anomaly_detector is None:
        from labmind.ml.anomaly_detector import LabAnomalyDetector
        _anomaly_detector = LabAnomalyDetector()
        try:
            _anomaly_detector.load_model()
        except FileNotFoundError:
            logger.warning("Anomaly detector model not found. Run training first.")
            _anomaly_detector = None
    return _anomaly_detector


def _get_sepsis_predictor():
    """Lazy-load the sepsis predictor model."""
    global _sepsis_predictor
    if _sepsis_predictor is None:
        from labmind.ml.sepsis_predictor import SepsisRiskPredictor
        _sepsis_predictor = SepsisRiskPredictor()
        try:
            _sepsis_predictor.load_model()
        except FileNotFoundError:
            logger.warning("Sepsis predictor model not found. Run training first.")
            _sepsis_predictor = None
    return _sepsis_predictor


def _get_trend_analyzer():
    """Lazy-load the trend analyzer."""
    global _trend_analyzer
    if _trend_analyzer is None:
        from labmind.ml.trend_analyzer import LabTrendAnalyzer
        _trend_analyzer = LabTrendAnalyzer()
    return _trend_analyzer


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def fetch_patient_lab_history(
    patient_id: str,
    test_filter: Optional[str] = None,
) -> str:
    """
    Fetch a patient's lab history with demographics and abnormality flags.

    Args:
        patient_id: Patient ID (e.g., "P001").
        test_filter: Optional test name to filter results (e.g., "Creatinine").
    """
    try:
        db = _get_db()
        patient = db.get_patient(patient_id)
        if not patient:
            return f"❌ Patient ID '{patient_id}' not found in the system."

        # Get results
        results = db.get_lab_results(patient_id, limit=100)

        if test_filter:
            results = [
                r for r in results
                if test_filter.lower() in r.test_name.lower()
            ]

        if not results:
            return (
                f"No lab results found for {patient.full_name} "
                f"({patient_id}){' matching filter: ' + test_filter if test_filter else ''}."
            )

        # Format output
        lines = [
            f"## Patient Lab History",
            f"**Patient:** {patient.full_name} | **MRN:** {patient.mrn} "
            f"| **Age:** {patient.age} | **Sex:** {patient.gender}",
            f"**Admitted:** {patient.admission_date or 'N/A'}",
            "",
            f"### Results ({len(results)} records)" + (
                f" — Filtered: {test_filter}" if test_filter else ""
            ),
            "",
            "| Date/Time | Test | Value | Unit | Range | Status |",
            "|-----------|------|-------|------|-------|--------|",
        ]

        for r in results[:50]:  # Cap display at 50
            flag = ""
            if r.is_critical:
                flag = " 🔴 CRITICAL"
            elif r.is_abnormal:
                flag = " ⚠️"

            lines.append(
                f"| {r.collection_datetime[:16]} | {r.test_name} "
                f"| {r.value:.2f} | {r.unit} "
                f"| {r.reference_range_low:.1f}–{r.reference_range_high:.1f} "
                f"| {r.status}{flag} |"
            )

        # Summary of abnormal values
        abnormal = [r for r in results if r.is_abnormal]
        critical = [r for r in results if r.is_critical]

        lines.extend([
            "",
            f"**Summary:** {len(abnormal)} abnormal values, "
            f"{len(critical)} critical values out of {len(results)} total.",
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in fetch_patient_lab_history")
        return f"❌ Error retrieving lab history: {e}"


@mcp.tool()
def run_lab_anomaly_detector(
    patient_id: str,
    wbc: Optional[float] = None,
    hemoglobin: Optional[float] = None,
    platelets: Optional[float] = None,
    creatinine: Optional[float] = None,
    lactate: Optional[float] = None,
    bun: Optional[float] = None,
    glucose: Optional[float] = None,
    potassium: Optional[float] = None,
) -> str:
    """
    Run ML-based anomaly detection on a patient's lab panel.

    Uses an Isolation Forest trained on historical data to detect
    multi-parameter anomalies that may not be apparent from individual values.

    Provide a patient_id to auto-fetch latest labs, or supply values directly.

    Args:
        patient_id: Patient ID to fetch labs from.
        wbc: White blood cell count (x10^3/µL). Optional override.
        hemoglobin: Hemoglobin (g/dL). Optional override.
        platelets: Platelet count (x10^3/µL). Optional override.
        creatinine: Creatinine (mg/dL). Optional override.
        lactate: Lactate (mmol/L). Optional override.
        bun: Blood urea nitrogen (mg/dL). Optional override.
        glucose: Glucose (mg/dL). Optional override.
        potassium: Potassium (mmol/L). Optional override.
    """
    try:
        detector = _get_anomaly_detector()
        if detector is None:
            return (
                "❌ Anomaly detector model not available. "
                "Run: python -m labmind.ml.train_models"
            )

        # Build lab values from patient data or direct input
        lab_values: dict[str, float] = {}
        overrides = {
            "wbc": wbc, "hemoglobin": hemoglobin, "platelets": platelets,
            "creatinine": creatinine, "lactate": lactate, "bun": bun,
            "glucose": glucose, "potassium": potassium,
        }

        db = _get_db()
        patient = db.get_patient(patient_id)

        if patient:
            panel = db.get_latest_lab_panel(patient_id)
            for test_name, result in panel.items():
                key = test_name.lower()
                if key in ANOMALY_FEATURES:
                    lab_values[key] = result.value

        # Apply overrides
        for key, val in overrides.items():
            if val is not None:
                lab_values[key] = val

        if not lab_values:
            return f"❌ No lab values available for patient '{patient_id}'."

        result = detector.detect(lab_values)

        # Format output
        lines = [
            f"## Lab Anomaly Detection Report",
        ]

        if patient:
            lines.append(
                f"**Patient:** {patient.full_name} ({patient_id}) "
                f"| **Age:** {patient.age} | **Sex:** {patient.gender}"
            )

        status_emoji = "🔴 ANOMALY DETECTED" if result.is_anomaly else "🟢 NORMAL"
        lines.extend([
            "",
            f"### Status: {status_emoji}",
            f"**Anomaly Score:** {result.anomaly_score:.4f} "
            f"| **Confidence:** {result.confidence:.1%}",
            "",
            "### Input Values & Z-Scores",
            "| Parameter | Value | Z-Score | Status |",
            "|-----------|-------|---------|--------|",
        ])

        for feat in ANOMALY_FEATURES:
            val = lab_values.get(feat, None)
            z = result.z_scores.get(feat, 0.0)
            flag = "🔴 Flagged" if feat in result.flagged_parameters else "✅"
            val_str = f"{val:.2f}" if val is not None else "N/A"
            lines.append(f"| {feat} | {val_str} | {z:+.3f} | {flag} |")

        if result.flagged_parameters:
            lines.extend([
                "",
                f"### Flagged Parameters",
                ", ".join(f"**{p}** (z={result.z_scores[p]:+.2f})"
                          for p in result.flagged_parameters),
            ])

        lines.extend(["", f"### Interpretation", result.explanation])

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in run_lab_anomaly_detector")
        return f"❌ Error running anomaly detection: {e}"


@mcp.tool()
def predict_sepsis_risk(
    patient_id: str,
    heart_rate: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    temperature: Optional[float] = None,
    respiratory_rate: Optional[float] = None,
) -> str:
    """
    Predict sepsis risk from a patient's latest labs and provided vitals.

    Combines lab values with vital signs to run an XGBoost classifier,
    returning risk probability, SHAP-based explanations, and clinical recommendations.

    Pass measured vitals whenever available. Omitted vitals are assumed normal
    (HR 80, SBP 120, Temp 37.0, RR 16), which can understate risk; the report
    marks them as assumed.

    Args:
        patient_id: Patient ID to fetch labs from (e.g., "P001").
        heart_rate: Heart rate in bpm.
        systolic_bp: Systolic blood pressure in mmHg.
        temperature: Body temperature in °C.
        respiratory_rate: Respiratory rate in breaths/min.
    """
    try:
        predictor = _get_sepsis_predictor()
        if predictor is None:
            return (
                "❌ Sepsis predictor model not available. "
                "Run: python -m labmind.ml.train_models"
            )

        db = _get_db()
        patient = db.get_patient(patient_id)
        if not patient:
            return f"❌ Patient ID '{patient_id}' not found."

        # Fetch latest labs
        panel = db.get_latest_lab_panel(patient_id)
        lab_values: dict[str, float] = {}
        for test_name, result in panel.items():
            key = test_name.lower()
            lab_values[key] = result.value

        # Add vitals, falling back to normal values for any not provided
        provided_vitals = {
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "temperature": temperature,
            "respiratory_rate": respiratory_rate,
        }
        assumed_vitals = [k for k, v in provided_vitals.items() if v is None]
        for key, val in provided_vitals.items():
            lab_values[key] = val if val is not None else DEFAULT_VITALS[key]

        result = predictor.predict(lab_values)

        # Format output
        risk_emoji = {
            "LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴", "CRITICAL": "⚠️"
        }.get(result.risk_level, "❓")

        lines = [
            f"## Sepsis Risk Assessment",
            f"**Patient:** {patient.full_name} ({patient_id}) "
            f"| **Age:** {patient.age} | **Sex:** {patient.gender}",
            f"**Assessment Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"### Risk Level: {risk_emoji} {result.risk_level}",
            f"**Probability:** {result.probability:.1%}",
            "",
            "### Input Parameters",
            "| Category | Parameter | Value |",
            "|----------|-----------|-------|",
        ]

        # Lab values
        for feat in ["wbc", "hemoglobin", "platelets", "creatinine",
                      "lactate", "bun", "glucose", "potassium"]:
            val = lab_values.get(feat, None)
            val_str = f"{val:.2f}" if val is not None else "N/A"
            lines.append(f"| Lab | {feat} | {val_str} |")

        # Vitals
        for key, fmt, unit in [
            ("heart_rate", ".0f", "bpm"),
            ("systolic_bp", ".0f", "mmHg"),
            ("temperature", ".1f", "°C"),
            ("respiratory_rate", ".0f", "/min"),
        ]:
            note = " *(assumed)*" if key in assumed_vitals else ""
            lines.append(
                f"| Vital | {key} | {lab_values[key]:{fmt}} {unit}{note} |"
            )

        if assumed_vitals:
            lines.extend([
                "",
                f"⚠️ **Vitals not provided — assumed normal:** "
                f"{', '.join(assumed_vitals)}. Risk may be understated; "
                f"re-run with measured vitals.",
            ])

        # SHAP explanation
        if result.top_risk_factors:
            lines.extend([
                "",
                "### Top Risk Factors (SHAP Analysis)",
                "| Factor | Value | SHAP Contribution | Direction |",
                "|--------|-------|--------------------|-----------|",
            ])
            for factor in result.top_risk_factors:
                lines.append(
                    f"| {factor['feature']} | {factor['value']} "
                    f"| {factor['shap_contribution']} | {factor['direction']} |"
                )

        lines.extend([
            "",
            "### Clinical Recommendation",
            result.recommendation,
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in predict_sepsis_risk")
        return f"❌ Error predicting sepsis risk: {e}"


@mcp.tool()
def analyze_lab_trend(
    patient_id: str,
    test_code: str,
    days: int = 180,
) -> str:
    """
    Analyze the temporal trend of a specific lab test for a patient.

    Uses linear regression to compute slope, direction, projected values,
    and clinical significance.

    Args:
        patient_id: Patient ID (e.g., "P001").
        test_code: LOINC code for the test (e.g., "2160-0" for Creatinine).
        days: Look-back period in days (default: 180).
    """
    try:
        db = _get_db()
        patient = db.get_patient(patient_id)
        if not patient:
            return f"❌ Patient ID '{patient_id}' not found."

        results = db.get_lab_trend(patient_id, test_code, days)
        if not results:
            code_info = LOINC_CODES.get(test_code, {})
            test_name = code_info.get("name", test_code)
            return (
                f"No results found for {test_name} (LOINC: {test_code}) "
                f"for patient {patient.full_name} in the last {days} days."
            )

        test_name = results[0].test_name
        values = [r.value for r in results]
        timestamps = [
            datetime.strptime(r.collection_datetime, "%Y-%m-%dT%H:%M:%S")
            for r in results
        ]

        analyzer = _get_trend_analyzer()
        trend = analyzer.analyze_trend(values, timestamps, test_name)

        # Format output
        direction_emoji = {
            "STABLE": "➡️",
            "INCREASING": "📈",
            "DECREASING": "📉",
            "RAPIDLY_INCREASING": "🔺",
            "RAPIDLY_DECREASING": "🔻",
            "INSUFFICIENT_DATA": "❓",
        }.get(trend.trend_direction, "❓")

        lines = [
            f"## Lab Trend Analysis: {test_name}",
            f"**Patient:** {patient.full_name} ({patient_id})",
            f"**LOINC Code:** {test_code} | **Period:** {days} days "
            f"| **Data Points:** {trend.n_datapoints}",
            "",
            f"### Trend: {direction_emoji} {trend.trend_direction}",
            f"**Slope:** {trend.slope_per_week:+.4f}/week "
            f"| **R²:** {trend.r_squared:.3f}",
            "",
            "### Historical Values",
            "| Date | Value | Unit | Status |",
            "|------|-------|------|--------|",
        ]

        for r in results:
            flag = " 🔴" if r.is_critical else (" ⚠️" if r.is_abnormal else "")
            lines.append(
                f"| {r.collection_datetime[:10]} "
                f"| {r.value:.2f} | {r.unit} | {r.status}{flag} |"
            )

        if trend.projected_values:
            lines.extend([
                "",
                "### Projections",
                "| Timeframe | Projected Value |",
                "|-----------|-----------------|",
            ])
            for label, val in trend.projected_values.items():
                lines.append(f"| {label.replace('_', ' ')} | {val:.2f} |")

        lines.extend([
            "",
            "### Clinical Assessment",
            f"**Significance:** {trend.clinical_significance}",
            "",
            trend.explanation,
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in analyze_lab_trend")
        return f"❌ Error analyzing lab trend: {e}"


@mcp.tool()
def search_patients(query: str) -> str:
    """
    Search for patients by name or MRN.

    Args:
        query: Search string to match against patient name or MRN.
    """
    try:
        db = _get_db()
        results = db.search_patients(query)

        if not results:
            return f"No patients found matching '{query}'."

        lines = [
            f"## Patient Search Results ({len(results)} found)",
            "",
            "| Patient ID | MRN | Name | Age | Sex | Admitted |",
            "|------------|-----|------|-----|-----|----------|",
        ]

        for p in results:
            lines.append(
                f"| {p.patient_id} | {p.mrn} | {p.full_name} "
                f"| {p.age} | {p.gender} | {p.admission_date or 'N/A'} |"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in search_patients")
        return f"❌ Error searching patients: {e}"


@mcp.tool()
def get_critical_alerts(patient_id: str) -> str:
    """
    Generate a prioritized list of critical alerts for a patient.

    Checks latest lab values for critical flags and runs anomaly detection
    to surface multi-parameter concerns.

    Args:
        patient_id: Patient ID (e.g., "P001").
    """
    try:
        db = _get_db()
        patient = db.get_patient(patient_id)
        if not patient:
            return f"❌ Patient ID '{patient_id}' not found."

        panel = db.get_latest_lab_panel(patient_id)
        if not panel:
            return f"No lab results available for patient {patient_id}."

        alerts: list[str] = []

        # Check for critical values
        for test_name, result in panel.items():
            if result.is_critical:
                alerts.append(
                    f"🔴 **CRITICAL** — {test_name}: {result.value:.2f} {result.unit} "
                    f"(ref: {result.reference_range_low:.1f}–{result.reference_range_high:.1f})"
                )
            elif result.is_abnormal:
                alerts.append(
                    f"⚠️ **Abnormal** — {test_name}: {result.value:.2f} {result.unit} "
                    f"({result.status})"
                )

        # Run anomaly detection
        detector = _get_anomaly_detector()
        if detector:
            lab_values = {}
            for test_name, result in panel.items():
                key = test_name.lower()
                if key in ANOMALY_FEATURES:
                    lab_values[key] = result.value

            if lab_values:
                anomaly = detector.detect(lab_values)
                if anomaly.is_anomaly:
                    flagged_str = ", ".join(anomaly.flagged_parameters[:3])
                    alerts.insert(0,
                        f"🧬 **ML Anomaly Alert** — Multi-parameter anomaly detected "
                        f"(score: {anomaly.anomaly_score:.4f}). "
                        f"Key parameters: {flagged_str or 'multivariate pattern'}"
                    )

        # Format output
        lines = [
            f"## Critical Alerts for {patient.full_name}",
            f"**Patient:** {patient_id} | **MRN:** {patient.mrn}",
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        if alerts:
            lines.append(f"### 🚨 {len(alerts)} Alert(s)")
            for i, alert in enumerate(alerts, 1):
                lines.append(f"{i}. {alert}")
        else:
            lines.append("### ✅ No Active Alerts")
            lines.append("All latest lab values are within normal limits.")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in get_critical_alerts")
        return f"❌ Error checking critical alerts: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource("lis://reference-ranges")
def get_reference_ranges() -> str:
    """
    Standard reference ranges for all laboratory tests.

    Returns structured reference data including gender-specific ranges
    and critical value thresholds.
    """
    try:
        db = _get_db()
        ranges = db.get_reference_ranges()

        lines = [
            "# Laboratory Reference Ranges",
            "",
            "| Test | Code | Unit | Normal Low | Normal High | "
            "Male Low | Male High | Female Low | Female High | "
            "Critical Low | Critical High |",
            "|------|------|------|------------|-------------|"
            "----------|-----------|------------|-------------|"
            "--------------|----------------|",
        ]

        for r in ranges:
            lines.append(
                f"| {r.test_name} | {r.test_code} | {r.unit} "
                f"| {r.range_low or ''} | {r.range_high or ''} "
                f"| {r.range_low_male or ''} | {r.range_high_male or ''} "
                f"| {r.range_low_female or ''} | {r.range_high_female or ''} "
                f"| {r.critical_low or ''} | {r.critical_high or ''} |"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error loading reference ranges: {e}"


@mcp.resource("lis://lab-codes")
def get_lab_codes() -> str:
    """
    LOINC code directory with test descriptions and categories.

    Searchable reference for mapping lab test codes to human-readable names.
    """
    lines = [
        "# LOINC Lab Code Directory",
        "",
        "| LOINC Code | Test Name | Unit | Category |",
        "|------------|-----------|------|----------|",
    ]

    for code, info in sorted(LOINC_CODES.items(), key=lambda x: x[1]["category"]):
        lines.append(
            f"| {code} | {info['name']} | {info['unit']} | {info['category']} |"
        )

    return "\n".join(lines)


@mcp.resource("lis://patient-roster")
def get_patient_roster() -> str:
    """
    Patient roster for discovery.

    Lists all patients with IDs, names, and MRNs for use with other tools.
    """
    try:
        db = _get_db()
        patients = db.get_all_patients()

        lines = [
            "# Patient Roster",
            "",
            f"Total patients: {len(patients)}",
            "",
            "| Patient ID | MRN | Name | Age | Sex |",
            "|------------|-----|------|-----|-----|",
        ]

        for p in patients:
            lines.append(
                f"| {p.patient_id} | {p.mrn} | {p.full_name} "
                f"| {p.age} | {p.gender} |"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error loading patient roster: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt()
def clinical_review(patient_id: str) -> str:
    """
    Generate a comprehensive clinical review workflow for a patient.

    Instructs the LLM to systematically fetch labs, run anomaly detection,
    check sepsis risk, and analyze key trends.

    Args:
        patient_id: The patient ID to review.
    """
    return f"""You are a clinical decision support assistant. Perform a comprehensive
review of patient {patient_id} by executing the following steps in order:

1. **Fetch Lab History**: Call `fetch_patient_lab_history` with patient_id="{patient_id}"
   to retrieve all available lab results. Note any abnormal or critical values.

2. **Check Critical Alerts**: Call `get_critical_alerts` with patient_id="{patient_id}"
   to get a prioritized alert list.

3. **Run Anomaly Detection**: Call `run_lab_anomaly_detector` with patient_id="{patient_id}"
   to check for multi-parameter anomalies that may not be obvious from individual values.

4. **Assess Sepsis Risk**: Call `predict_sepsis_risk` with patient_id="{patient_id}"
   and reasonable vital sign estimates. Report the risk level and key contributing factors.

5. **Trend Analysis**: For any abnormal lab values identified in step 1, call
   `analyze_lab_trend` with the appropriate test_code to assess whether values
   are stable, improving, or worsening.

6. **Synthesize**: Combine all findings into a structured clinical summary with:
   - Patient demographics
   - Key abnormal findings
   - ML-based risk assessments
   - Trend trajectories for concerning values
   - Prioritized clinical recommendations

Present your findings in a clear, structured format suitable for clinical handoff."""


@mcp.prompt()
def shift_handoff(patient_ids: str) -> str:
    """
    Generate a shift handoff report for multiple patients.

    Args:
        patient_ids: Comma-separated patient IDs (e.g., "P001,P002,P003").
    """
    ids = [pid.strip() for pid in patient_ids.split(",")]
    patient_list = "\n".join(f"   - {pid}" for pid in ids)

    return f"""You are preparing a nursing/physician shift handoff report.
For each of the following patients, provide a concise clinical summary:

Patients:
{patient_list}

For EACH patient:
1. Call `fetch_patient_lab_history` to get their latest labs
2. Call `get_critical_alerts` to check for any active alerts
3. Call `predict_sepsis_risk` to assess infection risk

Then compile a handoff report with this structure for each patient:

### [Patient Name] ({'{patient_id}'})
- **Status**: Stable / Monitoring / Critical
- **Key Labs**: List the 3-5 most relevant recent results
- **Active Alerts**: Any critical or abnormal flags
- **Risk Assessment**: Sepsis risk level
- **Action Items**: Pending tasks or follow-ups

End with an overall unit summary highlighting which patients need
the most urgent attention during this shift."""


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    """Run the LabMind-MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
