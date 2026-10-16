"""
Synthetic clinical data generator and database seeder for LabMind-MCP.

Generates realistic patients, lab results with medically plausible distributions,
and populates a SQLite database for development and demonstration.

Usage:
    python -m labmind.data.seed_database
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from labmind.config import DB_PATH

# Deterministic seeding
random.seed(42)
np.random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────

FIRST_NAMES_M = [
    "James", "Robert", "John", "Michael", "David", "William", "Richard",
    "Joseph", "Thomas", "Daniel", "Matthew", "Anthony", "Mark", "Steven",
    "Paul", "Andrew", "Kenneth", "George", "Edward", "Brian",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Dorothy",
    "Sandra", "Ashley", "Emily", "Donna", "Michelle", "Carol",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson",
]

# Reference ranges: (test_code, name, unit, low, high, low_m, high_m, low_f, high_f, crit_low, crit_high)
REFERENCE_RANGES = [
    ("6690-2",  "WBC",           "x10^3/µL", 4.5, 11.0, None, None, None, None, 2.0, 30.0),
    ("718-7",   "Hemoglobin",    "g/dL",     12.0, 17.5, 13.8, 17.2, 12.1, 15.1, 7.0, 20.0),
    ("4544-3",  "Hematocrit",    "%",        36.0, 52.0, 40.7, 50.3, 36.1, 44.3, 20.0, 60.0),
    ("777-3",   "Platelets",     "x10^3/µL", 150.0, 450.0, None, None, None, None, 50.0, 1000.0),
    ("789-8",   "RBC",           "x10^6/µL", 4.0, 6.0, 4.7, 6.1, 4.2, 5.4, 2.5, 8.0),
    ("787-2",   "MCV",           "fL",       80.0, 100.0, None, None, None, None, 60.0, 120.0),
    ("785-6",   "MCH",           "pg",       27.0, 33.0, None, None, None, None, 20.0, 40.0),
    ("786-4",   "MCHC",          "g/dL",     32.0, 36.0, None, None, None, None, 28.0, 40.0),
    ("2160-0",  "Creatinine",    "mg/dL",    0.74, 1.35, 0.74, 1.35, 0.59, 1.04, 0.4, 10.0),
    ("3094-0",  "BUN",           "mg/dL",    7.0, 20.0, None, None, None, None, 2.0, 100.0),
    ("2345-7",  "Glucose",       "mg/dL",    70.0, 100.0, None, None, None, None, 40.0, 500.0),
    ("2951-2",  "Sodium",        "mmol/L",   136.0, 145.0, None, None, None, None, 120.0, 160.0),
    ("2823-3",  "Potassium",     "mmol/L",   3.5, 5.0, None, None, None, None, 2.5, 6.5),
    ("2075-0",  "Chloride",      "mmol/L",   98.0, 106.0, None, None, None, None, 80.0, 120.0),
    ("2028-9",  "CO2",           "mmol/L",   23.0, 29.0, None, None, None, None, 10.0, 40.0),
    ("17861-6", "Calcium",       "mg/dL",    8.5, 10.5, None, None, None, None, 6.0, 14.0),
    ("2524-7",  "Lactate",       "mmol/L",   0.5, 2.2, None, None, None, None, 0.0, 8.0),
    ("33959-8", "Procalcitonin", "ng/mL",    0.0, 0.1, None, None, None, None, None, 10.0),
    ("1988-5",  "CRP",           "mg/L",     0.0, 5.0, None, None, None, None, None, 200.0),
]

# Normal distributions for healthy patients: (test_code, mean, std)
NORMAL_DISTRIBUTIONS: dict[str, tuple[float, float]] = {
    "6690-2":  (7.5, 1.5),
    "718-7":   (14.5, 1.5),
    "4544-3":  (43.0, 3.5),
    "777-3":   (250.0, 50.0),
    "789-8":   (5.0, 0.5),
    "787-2":   (90.0, 5.0),
    "785-6":   (30.0, 1.5),
    "786-4":   (34.0, 1.0),
    "2160-0":  (1.0, 0.15),
    "3094-0":  (14.0, 3.0),
    "2345-7":  (90.0, 10.0),
    "2951-2":  (140.0, 2.0),
    "2823-3":  (4.2, 0.3),
    "2075-0":  (102.0, 2.0),
    "2028-9":  (26.0, 1.5),
    "17861-6": (9.5, 0.4),
    "2524-7":  (1.2, 0.4),
    "33959-8": (0.04, 0.02),
    "1988-5":  (2.0, 1.5),
}


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create the database schema."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            mrn TEXT UNIQUE NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            gender TEXT NOT NULL CHECK(gender IN ('M', 'F')),
            admission_date TEXT
        );

        CREATE TABLE IF NOT EXISTS lab_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            test_code TEXT NOT NULL,
            test_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            reference_range_low REAL,
            reference_range_high REAL,
            collection_datetime TEXT NOT NULL,
            is_critical INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );

        CREATE TABLE IF NOT EXISTS reference_ranges (
            test_code TEXT PRIMARY KEY,
            test_name TEXT NOT NULL,
            unit TEXT NOT NULL,
            range_low REAL,
            range_high REAL,
            range_low_male REAL,
            range_high_male REAL,
            range_low_female REAL,
            range_high_female REAL,
            critical_low REAL,
            critical_high REAL
        );

        CREATE INDEX IF NOT EXISTS idx_lab_patient ON lab_results(patient_id);
        CREATE INDEX IF NOT EXISTS idx_lab_test ON lab_results(test_code);
        CREATE INDEX IF NOT EXISTS idx_lab_datetime ON lab_results(collection_datetime);
    """)


def _seed_reference_ranges(conn: sqlite3.Connection) -> None:
    """Populate the reference_ranges table."""
    conn.executemany(
        """INSERT OR REPLACE INTO reference_ranges
           (test_code, test_name, unit, range_low, range_high,
            range_low_male, range_high_male, range_low_female, range_high_female,
            critical_low, critical_high)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        REFERENCE_RANGES,
    )


def _generate_collection_times(
    num_timepoints: int, reference_time: datetime
) -> list[datetime]:
    """
    Generate strictly increasing collection times ending before reference_time.

    Samples are 1–5 days apart, so the series always reaches the present
    and stays inside the default trend look-back window.
    """
    day_offsets = [0]
    for _ in range(num_timepoints - 1):
        day_offsets.append(day_offsets[-1] + random.randint(1, 5))

    # Last collection lands on the day before reference_time
    start = (reference_time - timedelta(days=day_offsets[-1] + 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return [
        start + timedelta(
            days=offset,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        for offset in day_offsets
    ]


def _generate_patient(
    idx: int, reference_time: datetime, admission_time: datetime
) -> tuple[str, str, str, str, str, str, str]:
    """Generate a single patient record."""
    patient_id = f"P{idx:03d}"
    mrn = f"MRN-{random.randint(100000, 999999)}"
    gender = random.choice(["M", "F"])

    if gender == "M":
        first_name = random.choice(FIRST_NAMES_M)
    else:
        first_name = random.choice(FIRST_NAMES_F)
    last_name = random.choice(LAST_NAMES)

    # Age between 18 and 90
    age = random.randint(18, 90)
    dob = reference_time - timedelta(days=age * 365 + random.randint(0, 364))
    dob_str = dob.strftime("%Y-%m-%d")

    # Admitted on the day of the first lab draw
    admit_str = admission_time.strftime("%Y-%m-%d")

    return (patient_id, mrn, first_name, last_name, dob_str, gender, admit_str)


def _generate_lab_value(
    test_code: str,
    is_sepsis: bool = False,
    is_aki: bool = False,
    time_step: int = 0,
    total_steps: int = 1,
) -> float:
    """
    Generate a lab value from medically realistic distributions.

    For sepsis/AKI patients, values progressively worsen over time_step.
    """
    mean, std = NORMAL_DISTRIBUTIONS[test_code]
    progression = time_step / max(total_steps - 1, 1)  # 0.0 → 1.0

    if is_sepsis:
        if test_code == "6690-2":      # WBC rises
            mean += 8.0 * progression
        elif test_code == "2524-7":    # Lactate rises
            mean += 4.0 * progression
        elif test_code == "777-3":     # Platelets drop
            mean -= 120.0 * progression
        elif test_code == "2160-0":    # Creatinine rises
            mean += 1.5 * progression
        elif test_code == "33959-8":   # Procalcitonin rises
            mean += 5.0 * progression
        elif test_code == "1988-5":    # CRP rises
            mean += 80.0 * progression

    if is_aki:
        if test_code == "2160-0":      # Creatinine rises sharply
            mean += 3.0 * progression
        elif test_code == "3094-0":    # BUN rises
            mean += 30.0 * progression
        elif test_code == "2823-3":    # Potassium rises
            mean += 1.5 * progression

    value = np.random.normal(mean, std)

    # Clamp to physiologically possible ranges
    ref = next((r for r in REFERENCE_RANGES if r[0] == test_code), None)
    if ref:
        crit_low = ref[9] if ref[9] is not None else ref[3] * 0.3
        crit_high = ref[10] if ref[10] is not None else ref[4] * 3.0
        value = max(crit_low * 0.8, min(value, crit_high * 1.2))

    return round(value, 2)


def _is_critical(test_code: str, value: float) -> bool:
    """Check whether a lab value is in the critical range."""
    ref = next((r for r in REFERENCE_RANGES if r[0] == test_code), None)
    if not ref:
        return False
    crit_low, crit_high = ref[9], ref[10]
    if crit_low is not None and value < crit_low:
        return True
    if crit_high is not None and value > crit_high:
        return True
    return False


def seed_database(
    db_path: Path | None = None,
    num_patients: int = 60,
    reference_time: datetime | None = None,
) -> None:
    """
    Create and populate the LabMind database with synthetic clinical data.

    Args:
        db_path: Path for the SQLite database. Defaults to config.DB_PATH.
        num_patients: Number of patients to generate (default 60).
        reference_time: "Now" for the generated timeline; all collections fall
            before it. Defaults to the current time.
    """
    if db_path is None:
        db_path = DB_PATH
    if reference_time is None:
        reference_time = datetime.now()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB for clean seed
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))

    try:
        _create_tables(conn)
        _seed_reference_ranges(conn)

        # ── Generate patients and lab results ─────────────────────────────
        patients = []
        # Assign ~10% sepsis, ~8% AKI, rest healthy
        sepsis_ids = set(range(1, int(num_patients * 0.10) + 1))
        aki_ids = set(range(
            int(num_patients * 0.10) + 1,
            int(num_patients * 0.18) + 1,
        ))

        all_test_codes = [r[0] for r in REFERENCE_RANGES]
        lab_rows: list[tuple] = []

        for idx in range(1, num_patients + 1):
            patient_id = f"P{idx:03d}"
            is_sepsis = idx in sepsis_ids
            is_aki = idx in aki_ids

            # Number of time points per patient
            if is_sepsis or is_aki:
                num_timepoints = random.randint(8, 15)
            else:
                num_timepoints = random.randint(3, 8)

            collection_times = _generate_collection_times(
                num_timepoints, reference_time
            )
            patients.append(
                _generate_patient(idx, reference_time, collection_times[0])
            )

            for t, collection_dt in enumerate(collection_times):
                dt_str = collection_dt.strftime("%Y-%m-%dT%H:%M:%S")

                for tc in all_test_codes:
                    value = _generate_lab_value(
                        tc,
                        is_sepsis=is_sepsis,
                        is_aki=is_aki,
                        time_step=t,
                        total_steps=num_timepoints,
                    )
                    ref = next(r for r in REFERENCE_RANGES if r[0] == tc)
                    ref_low, ref_high = ref[3], ref[4]
                    is_crit = 1 if _is_critical(tc, value) else 0

                    lab_rows.append((
                        patient_id, tc, ref[1], value, ref[2],
                        ref_low, ref_high, dt_str, is_crit,
                    ))

        conn.executemany(
            """INSERT INTO patients
               (patient_id, mrn, first_name, last_name, date_of_birth, gender, admission_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            patients,
        )
        conn.executemany(
            """INSERT INTO lab_results
               (patient_id, test_code, test_name, value, unit,
                reference_range_low, reference_range_high,
                collection_datetime, is_critical)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            lab_rows,
        )

        conn.commit()
        total_labs = len(lab_rows)
        critical_count = sum(1 for r in lab_rows if r[8] == 1)

        print(f"✅ Database seeded at: {db_path}")
        print(f"   Patients: {num_patients}")
        print(f"   Lab results: {total_labs}")
        print(f"   Critical values: {critical_count}")
        print(f"   Sepsis trajectories: {len(sepsis_ids)} patients")
        print(f"   AKI trajectories: {len(aki_ids)} patients")

    finally:
        conn.close()


if __name__ == "__main__":
    seed_database()
