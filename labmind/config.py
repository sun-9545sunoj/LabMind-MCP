"""
Configuration management for LabMind-MCP.

Centralizes paths, model hyperparameters, and server settings.
Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "labmind.db"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"
SEPSIS_MODEL_PATH = MODELS_DIR / "sepsis_predictor.joblib"

# ─── Model Hyperparameters ────────────────────────────────────────────────────

ANOMALY_CONTAMINATION = float(os.getenv("LABMIND_ANOMALY_CONTAMINATION", "0.05"))
ANOMALY_RANDOM_STATE = int(os.getenv("LABMIND_ANOMALY_SEED", "42"))

SEPSIS_N_ESTIMATORS = int(os.getenv("LABMIND_SEPSIS_ESTIMATORS", "100"))
SEPSIS_MAX_DEPTH = int(os.getenv("LABMIND_SEPSIS_DEPTH", "4"))
SEPSIS_LEARNING_RATE = float(os.getenv("LABMIND_SEPSIS_LR", "0.1"))

# ─── Feature Definitions ─────────────────────────────────────────────────────

ANOMALY_FEATURES = [
    "wbc", "hemoglobin", "platelets", "creatinine",
    "lactate", "bun", "glucose", "potassium",
]

SEPSIS_LAB_FEATURES = [
    "wbc", "hemoglobin", "platelets", "creatinine",
    "lactate", "bun", "glucose", "potassium",
]

SEPSIS_VITALS_FEATURES = [
    "heart_rate", "systolic_bp", "temperature", "respiratory_rate",
]

SEPSIS_ALL_FEATURES = SEPSIS_LAB_FEATURES + SEPSIS_VITALS_FEATURES

# ─── Risk Thresholds ─────────────────────────────────────────────────────────

SEPSIS_RISK_THRESHOLDS = {
    "LOW": 0.2,
    "MODERATE": 0.5,
    "HIGH": 0.8,
    # Anything ≥ 0.8 → CRITICAL
}

# ─── Server Settings ─────────────────────────────────────────────────────────

SERVER_NAME = "LabMind-MCP"
SERVER_VERSION = "0.1.0"

# ─── Lab Test LOINC Codes ────────────────────────────────────────────────────

LOINC_CODES: dict[str, dict[str, str]] = {
    "6690-2":  {"name": "WBC",           "unit": "x10^3/µL",  "category": "CBC"},
    "718-7":   {"name": "Hemoglobin",    "unit": "g/dL",      "category": "CBC"},
    "4544-3":  {"name": "Hematocrit",    "unit": "%",         "category": "CBC"},
    "777-3":   {"name": "Platelets",     "unit": "x10^3/µL",  "category": "CBC"},
    "789-8":   {"name": "RBC",           "unit": "x10^6/µL",  "category": "CBC"},
    "787-2":   {"name": "MCV",           "unit": "fL",        "category": "CBC"},
    "785-6":   {"name": "MCH",           "unit": "pg",        "category": "CBC"},
    "786-4":   {"name": "MCHC",          "unit": "g/dL",      "category": "CBC"},
    "2160-0":  {"name": "Creatinine",    "unit": "mg/dL",     "category": "Metabolic"},
    "3094-0":  {"name": "BUN",           "unit": "mg/dL",     "category": "Metabolic"},
    "2345-7":  {"name": "Glucose",       "unit": "mg/dL",     "category": "Metabolic"},
    "2951-2":  {"name": "Sodium",        "unit": "mmol/L",    "category": "Metabolic"},
    "2823-3":  {"name": "Potassium",     "unit": "mmol/L",    "category": "Metabolic"},
    "2075-0":  {"name": "Chloride",      "unit": "mmol/L",    "category": "Metabolic"},
    "2028-9":  {"name": "CO2",           "unit": "mmol/L",    "category": "Metabolic"},
    "17861-6": {"name": "Calcium",       "unit": "mg/dL",     "category": "Metabolic"},
    "2524-7":  {"name": "Lactate",       "unit": "mmol/L",    "category": "Sepsis"},
    "33959-8": {"name": "Procalcitonin", "unit": "ng/mL",     "category": "Sepsis"},
    "1988-5":  {"name": "CRP",           "unit": "mg/L",      "category": "Sepsis"},
}
