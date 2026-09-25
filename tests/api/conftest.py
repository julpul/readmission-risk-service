"""Fixtures for API tests.

The real artifact needs the full dataset, so tests use a tiny pipeline fitted on
synthetic rows instead. It is the REAL feature pipeline (build_full_pipeline) -
only the training data is fake - so the API is exercised end to end.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from readmission.api.app import create_app
from readmission.api.schemas import Encounter
from readmission.data.constants import AGE_CLASSES
from readmission.features.pipeline import build_full_pipeline
from readmission.models.artifact import ModelArtifact, save_artifact
from readmission.models.build_artifact import required_input_columns

THRESHOLD = 0.6
MEDIUM_THRESHOLD = 0.4


@pytest.fixture
def valid_encounter() -> dict:
    """A minimal valid request body; medications not listed default to "No"."""
    return {
        "age": "[70-80)",
        "race": "Caucasian",
        "gender": "Female",
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "number_inpatient": 1,
        "number_emergency": 0,
        "medical_specialty": "InternalMedicine",
        "diag_1": "428",
        "diag_2": "250.02",
        "diag_3": "401",
        "change": "Ch",
        "insulin": "Up",
        "metformin": "Steady",
    }


def synthetic_training_data(n_rows: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows = [
        Encounter(
            age=str(rng.choice(AGE_CLASSES)),
            race=str(rng.choice(["Caucasian", "AfricanAmerican"])),
            gender=str(rng.choice(["Male", "Female"])),
            time_in_hospital=int(rng.integers(1, 15)),
            num_lab_procedures=int(rng.integers(1, 100)),
            number_inpatient=int(rng.integers(0, 4)),
            number_emergency=int(rng.integers(0, 3)),
            medical_specialty=str(rng.choice(["InternalMedicine", "Cardiology"])),
            diag_1=str(rng.choice(["428", "250.83", "V45"])),
            diag_2=str(rng.choice(["401", "786"])),
            diag_3=str(rng.choice(["414", "250"])),
            change=str(rng.choice(["No", "Ch"])),
            insulin=str(rng.choice(["No", "Steady", "Up", "Down"])),
            metformin=str(rng.choice(["No", "Steady"])),
        ).to_row()
        for _ in range(n_rows)
    ]
    X = pd.DataFrame(rows)
    # Identifiers are present in real training data; the pipeline must drop them.
    X["encounter_id"] = range(n_rows)
    X["patient_nbr"] = range(1000, 1000 + n_rows)
    # Label loosely tied to prior admissions, so the model learns *something*.
    y = pd.Series((X["number_inpatient"] >= 2).astype(int))
    return X, y


@pytest.fixture(scope="session")
def dummy_artifact() -> ModelArtifact:
    X, y = synthetic_training_data()
    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    return ModelArtifact(
        pipeline=pipeline,
        threshold=THRESHOLD,
        medium_threshold=MEDIUM_THRESHOLD,
        model_version="test-0",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        input_columns=required_input_columns(pipeline),
        metrics={"pr_auc": 0.5},
        params={"max_iter": 1000},
    )


@pytest.fixture
def artifact_path(tmp_path, dummy_artifact):
    return save_artifact(dummy_artifact, tmp_path / "model.joblib")


@pytest.fixture
def client(artifact_path):
    # `with` runs the lifespan, i.e. the real startup model loading
    with TestClient(create_app(artifact_path)) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "does-not-exist.joblib")) as test_client:
        yield test_client
