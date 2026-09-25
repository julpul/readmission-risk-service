from dataclasses import replace

import joblib
import pytest
from fastapi.testclient import TestClient

from readmission.api.app import create_app
from readmission.models.artifact import load_artifact, save_artifact


def test_save_load_roundtrip(tmp_path, dummy_artifact):
    path = save_artifact(dummy_artifact, tmp_path / "nested" / "model.joblib")
    loaded = load_artifact(path)

    assert loaded.model_version == dummy_artifact.model_version
    assert loaded.threshold == dummy_artifact.threshold
    assert loaded.input_columns == dummy_artifact.input_columns


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope.joblib")


def test_load_rejects_non_artifact(tmp_path, dummy_artifact):
    # e.g. someone dumped the bare Pipeline - no threshold, no version
    path = tmp_path / "bare.joblib"
    joblib.dump(dummy_artifact.pipeline, path)

    with pytest.raises(TypeError, match="expected ModelArtifact"):
        load_artifact(path)


def test_api_refuses_artifact_needing_unknown_columns(tmp_path, dummy_artifact):
    # A model trained on a column the request schema cannot supply must not be served.
    broken = replace(dummy_artifact, input_columns=[*dummy_artifact.input_columns, "weight"])
    path = save_artifact(broken, tmp_path / "model.joblib")

    with TestClient(create_app(path)) as client:
        assert client.get("/health").json()["model_loaded"] is False
        detail = client.get("/model/info").json()["detail"]
        assert "weight" in detail
