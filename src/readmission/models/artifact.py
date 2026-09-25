"""Model artifact: the fitted pipeline plus everything needed to serve it.

A bare pickled Pipeline is not enough for a service. The API also needs the
decision thresholds, the raw columns the pipeline expects, and some provenance
(when it was trained, on what, with which metrics). All of it travels together
in one joblib file so the model and its threshold can never get out of sync.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.pipeline import Pipeline

DEFAULT_ARTIFACT_PATH = Path("artifacts") / "model.joblib"


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    # Operating point used for the "high" band (recall_at_precision on OOF data).
    threshold: float
    # Lower operating point used for the "medium" band (F1-optimal on OOF data).
    medium_threshold: float
    model_version: str
    trained_at: datetime
    input_columns: list[str]
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    data_sha256: str | None = None
    sklearn_version: str = sklearn.__version__


def save_artifact(artifact: ModelArtifact, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path) -> ModelArtifact:
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError(f"expected ModelArtifact, got {type(artifact).__name__}")
    return artifact
