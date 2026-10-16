"""Invariants of the synthetic data generator."""

import sqlite3
from datetime import datetime, timedelta

import pytest

from tests.conftest import NUM_PATIENTS


@pytest.fixture
def conn(db_path):
    connection = sqlite3.connect(str(db_path))
    yield connection
    connection.close()


def test_patient_count(conn):
    assert conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == NUM_PATIENTS


def test_collections_strictly_increase_in_generation_order(conn):
    out_of_order = conn.execute(
        """SELECT COUNT(*) FROM lab_results a JOIN lab_results b
             ON a.patient_id = b.patient_id AND a.test_code = b.test_code
            AND b.result_id > a.result_id
            AND b.collection_datetime <= a.collection_datetime"""
    ).fetchone()[0]
    assert out_of_order == 0


def test_collections_end_before_reference_time(conn, reference_time):
    latest = conn.execute(
        "SELECT MAX(collection_datetime) FROM lab_results"
    ).fetchone()[0]
    latest_dt = datetime.strptime(latest, "%Y-%m-%dT%H:%M:%S")
    assert latest_dt < reference_time
    # Most recent data is recent, so default 180-day trend windows include it
    assert reference_time - latest_dt < timedelta(days=2)


def test_admission_date_is_first_collection(conn):
    mismatches = conn.execute(
        """SELECT COUNT(*) FROM patients p
            WHERE admission_date != (
                SELECT substr(MIN(collection_datetime), 1, 10)
                  FROM lab_results l WHERE l.patient_id = p.patient_id)"""
    ).fetchone()[0]
    assert mismatches == 0


def test_sepsis_trajectory_lactate_rises(db):
    # P001 is in the first 10% of patients, which follow sepsis trajectories
    trend = db.get_lab_trend("P001", "2524-7")
    assert len(trend) >= 8
    first_half = [r.value for r in trend[: len(trend) // 2]]
    second_half = [r.value for r in trend[len(trend) // 2 :]]
    assert sum(second_half) / len(second_half) > sum(first_half) / len(first_half)
