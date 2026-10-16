"""Shared fixtures: a freshly seeded temp database and in-memory trained models.

Nothing here touches labmind/data/labmind.db or labmind/ml/models/*.joblib.
"""

from datetime import datetime

import pytest

from labmind.data.seed_database import seed_database
from labmind.db.database import LabDatabase
from labmind.ml.anomaly_detector import LabAnomalyDetector
from labmind.ml.sepsis_predictor import (
    SepsisRiskPredictor,
    generate_sepsis_training_data,
)
from labmind.ml.train_models import build_anomaly_training_frame

NUM_PATIENTS = 60


@pytest.fixture(scope="session")
def reference_time() -> datetime:
    return datetime.now().replace(microsecond=0)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory, reference_time):
    path = tmp_path_factory.mktemp("lis") / "labmind.db"
    seed_database(
        db_path=path, num_patients=NUM_PATIENTS, reference_time=reference_time
    )
    return path


@pytest.fixture(scope="session")
def db(db_path):
    database = LabDatabase(db_path)
    yield database
    database.close()


@pytest.fixture(scope="session")
def anomaly_detector(db):
    detector = LabAnomalyDetector()
    detector.train(build_anomaly_training_frame(db))
    return detector


@pytest.fixture(scope="session")
def sepsis_predictor():
    X, y = generate_sepsis_training_data(n_samples=2000, sepsis_ratio=0.25)
    predictor = SepsisRiskPredictor()
    predictor.train(X, y)
    return predictor


@pytest.fixture
def server(monkeypatch, db, anomaly_detector, sepsis_predictor):
    """The server module wired to the temp DB and in-memory models."""
    from labmind import server as server_module

    monkeypatch.setattr(server_module, "_db", db)
    monkeypatch.setattr(server_module, "_anomaly_detector", anomaly_detector)
    monkeypatch.setattr(server_module, "_sepsis_predictor", sepsis_predictor)
    return server_module
