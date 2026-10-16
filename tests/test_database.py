"""LabDatabase query behavior."""

import pytest

from labmind.db.database import LabDatabase


def test_missing_database_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LabDatabase(tmp_path / "nope.db")


def test_get_patient(db):
    patient = db.get_patient("P001")
    assert patient is not None
    assert patient.patient_id == "P001"
    assert db.get_patient("P999") is None


def test_search_patients_by_mrn(db):
    patient = db.get_patient("P002")
    results = db.search_patients(patient.mrn)
    assert [p.patient_id for p in results] == ["P002"]


def test_lab_results_newest_first(db):
    results = db.get_lab_results("P001", limit=100)
    times = [r.collection_datetime for r in results]
    assert times == sorted(times, reverse=True)


def test_latest_panel_has_most_recent_value_per_test(db):
    panel = db.get_latest_lab_panel("P001")
    assert "Creatinine" in panel
    history = db.get_lab_results("P001", test_code="2160-0", limit=100)
    assert panel["Creatinine"].collection_datetime == history[0].collection_datetime


def test_lab_trend_is_chronological(db):
    trend = db.get_lab_trend("P001", "2160-0")
    times = [r.collection_datetime for r in trend]
    assert len(times) >= 2
    assert times == sorted(times)


def test_lab_trend_respects_window(db):
    full = db.get_lab_trend("P001", "2160-0", days=180)
    recent = db.get_lab_trend("P001", "2160-0", days=3)
    assert 0 < len(recent) < len(full)
