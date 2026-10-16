"""MCP tools, resources, and prompts.

Tools return markdown strings and catch their own exceptions, so a failure
shows up as a "❌" message rather than a raise — every test checks for that.
"""

import pytest

ERROR_MARK = "❌"


def test_fetch_patient_lab_history(server):
    out = server.fetch_patient_lab_history("P001")
    assert ERROR_MARK not in out
    assert "## Patient Lab History" in out


def test_fetch_patient_lab_history_filter(server):
    out = server.fetch_patient_lab_history("P001", test_filter="creatinine")
    assert "Creatinine" in out
    assert "| Lactate" not in out


def test_run_lab_anomaly_detector(server):
    out = server.run_lab_anomaly_detector("P001")
    assert ERROR_MARK not in out
    assert "Anomaly Score" in out


def test_predict_sepsis_risk_with_vitals(server):
    out = server.predict_sepsis_risk(
        "P001", heart_rate=110, systolic_bp=85, temperature=38.9,
        respiratory_rate=24,
    )
    assert ERROR_MARK not in out
    assert "### Risk Level:" in out
    assert "assumed" not in out


def test_predict_sepsis_risk_marks_assumed_vitals(server):
    out = server.predict_sepsis_risk("P001", heart_rate=110)
    assert ERROR_MARK not in out
    assert "| heart_rate | 110 bpm |" in out
    assert "| systolic_bp | 120 mmHg *(assumed)* |" in out
    assert "assumed normal:** systolic_bp, temperature, respiratory_rate" in out


def test_analyze_lab_trend(server):
    out = server.analyze_lab_trend("P001", "2160-0")
    assert ERROR_MARK not in out
    assert "### Trend:" in out
    assert "No results found" not in out


def test_search_patients(server, db):
    name = db.get_patient("P001").last_name
    out = server.search_patients(name)
    assert "P001" in out


def test_get_critical_alerts(server):
    out = server.get_critical_alerts("P001")
    assert ERROR_MARK not in out
    assert "Critical Alerts" in out


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.fetch_patient_lab_history("P999"),
        lambda s: s.predict_sepsis_risk("P999"),
        lambda s: s.analyze_lab_trend("P999", "2160-0"),
        lambda s: s.get_critical_alerts("P999"),
    ],
)
def test_unknown_patient(server, call):
    assert "not found" in call(server)


def test_resources(server):
    assert "Creatinine" in server.get_reference_ranges()
    assert "2160-0" in server.get_lab_codes()
    assert "P060" in server.get_patient_roster()


def test_prompts():
    from labmind import server

    assert "P001" in server.clinical_review("P001")
    handoff = server.shift_handoff("P001, P002")
    assert "- P001" in handoff and "- P002" in handoff
