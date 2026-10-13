"""
Database access layer for LabMind-MCP.

Provides typed access to the SQLite laboratory information database
with patient records, lab results, and reference ranges.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from labmind.config import DB_PATH

logger = logging.getLogger(__name__)


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Patient:
    """Patient demographic record."""

    patient_id: str
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    admission_date: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self) -> int:
        dob = datetime.strptime(self.date_of_birth, "%Y-%m-%d")
        today = datetime.now()
        return today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )


@dataclass
class LabResult:
    """Single lab test result."""

    result_id: int
    patient_id: str
    test_code: str
    test_name: str
    value: float
    unit: str
    reference_range_low: float
    reference_range_high: float
    collection_datetime: str
    is_critical: bool

    @property
    def is_abnormal(self) -> bool:
        return (
            self.value < self.reference_range_low
            or self.value > self.reference_range_high
        )

    @property
    def status(self) -> str:
        if self.is_critical:
            return "CRITICAL"
        elif self.value < self.reference_range_low:
            return "LOW"
        elif self.value > self.reference_range_high:
            return "HIGH"
        return "NORMAL"


@dataclass
class ReferenceRange:
    """Reference range for a lab test."""

    test_code: str
    test_name: str
    unit: str
    range_low: float
    range_high: float
    range_low_male: Optional[float] = None
    range_high_male: Optional[float] = None
    range_low_female: Optional[float] = None
    range_high_female: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None


# ─── Database Access ──────────────────────────────────────────────────────────

class LabDatabase:
    """
    SQLite database access layer for the Laboratory Information System.

    Provides typed query methods for patients, lab results, and reference ranges.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DB_PATH
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self._db_path}. "
                f"Run: python -m labmind.data.seed_database"
            )
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        logger.info("Connected to database: %s", self._db_path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Patient Queries ───────────────────────────────────────────────────

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Fetch a single patient by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Patient(**dict(row))

    def search_patients(self, query: str) -> list[Patient]:
        """Search patients by name or MRN (case-insensitive)."""
        like_query = f"%{query}%"
        cursor = self._conn.execute(
            """SELECT * FROM patients
               WHERE first_name LIKE ? OR last_name LIKE ? OR mrn LIKE ?
               ORDER BY last_name, first_name""",
            (like_query, like_query, like_query),
        )
        return [Patient(**dict(row)) for row in cursor.fetchall()]

    def get_all_patients(self) -> list[Patient]:
        """Return all patients."""
        cursor = self._conn.execute(
            "SELECT * FROM patients ORDER BY patient_id"
        )
        return [Patient(**dict(row)) for row in cursor.fetchall()]

    # ── Lab Result Queries ────────────────────────────────────────────────

    def get_lab_results(
        self,
        patient_id: str,
        test_code: Optional[str] = None,
        limit: int = 50,
    ) -> list[LabResult]:
        """
        Fetch lab results for a patient.

        Args:
            patient_id: Patient identifier.
            test_code: Optional LOINC code to filter by.
            limit: Maximum number of results.
        """
        if test_code:
            cursor = self._conn.execute(
                """SELECT * FROM lab_results
                   WHERE patient_id = ? AND test_code = ?
                   ORDER BY collection_datetime DESC
                   LIMIT ?""",
                (patient_id, test_code, limit),
            )
        else:
            cursor = self._conn.execute(
                """SELECT * FROM lab_results
                   WHERE patient_id = ?
                   ORDER BY collection_datetime DESC
                   LIMIT ?""",
                (patient_id, limit),
            )
        return [
            LabResult(**{**dict(row), "is_critical": bool(row["is_critical"])})
            for row in cursor.fetchall()
        ]

    def get_latest_lab_panel(self, patient_id: str) -> dict[str, LabResult]:
        """
        Get the most recent result for each test for a patient.

        Returns:
            Dict mapping test_name -> most recent LabResult.
        """
        cursor = self._conn.execute(
            """SELECT lr.* FROM lab_results lr
               INNER JOIN (
                   SELECT test_code, MAX(collection_datetime) as max_dt
                   FROM lab_results
                   WHERE patient_id = ?
                   GROUP BY test_code
               ) latest ON lr.test_code = latest.test_code
                       AND lr.collection_datetime = latest.max_dt
               WHERE lr.patient_id = ?""",
            (patient_id, patient_id),
        )
        results: dict[str, LabResult] = {}
        for row in cursor.fetchall():
            lr = LabResult(**{**dict(row), "is_critical": bool(row["is_critical"])})
            results[lr.test_name] = lr
        return results

    def get_lab_trend(
        self,
        patient_id: str,
        test_code: str,
        days: int = 180,
    ) -> list[LabResult]:
        """
        Get chronological lab results for trend analysis.

        Args:
            patient_id: Patient identifier.
            test_code: LOINC code for the test.
            days: Look-back period in days.
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        cursor = self._conn.execute(
            """SELECT * FROM lab_results
               WHERE patient_id = ? AND test_code = ?
                 AND collection_datetime >= ?
               ORDER BY collection_datetime ASC""",
            (patient_id, test_code, cutoff),
        )
        return [
            LabResult(**{**dict(row), "is_critical": bool(row["is_critical"])})
            for row in cursor.fetchall()
        ]

    # ── Reference Range Queries ───────────────────────────────────────────

    def get_reference_ranges(
        self, test_code: Optional[str] = None
    ) -> list[ReferenceRange]:
        """Fetch reference ranges, optionally filtered by test code."""
        if test_code:
            cursor = self._conn.execute(
                "SELECT * FROM reference_ranges WHERE test_code = ?",
                (test_code,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM reference_ranges ORDER BY test_name"
            )
        return [ReferenceRange(**dict(row)) for row in cursor.fetchall()]
