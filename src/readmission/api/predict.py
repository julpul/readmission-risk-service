"""Glue between validated requests and the fitted pipeline. No FastAPI here."""

import pandas as pd

from readmission.api.schemas import Encounter, Prediction, RiskBand
from readmission.models.artifact import ModelArtifact


def risk_band(probability: float, threshold: float, medium_threshold: float) -> RiskBand:
    # >= to match compute_metrics(): a score equal to the threshold is flagged
    if probability >= threshold:
        return "high"
    if probability >= medium_threshold:
        return "medium"
    return "low"


def missing_columns(artifact: ModelArtifact) -> list[str]:
    """Columns the model needs that the request schema cannot provide."""
    return sorted(set(artifact.input_columns) - Encounter.raw_columns())


def predict(artifact: ModelArtifact, encounters: list[Encounter]) -> list[Prediction]:
    frame = pd.DataFrame([e.to_row() for e in encounters])
    probabilities = artifact.pipeline.predict_proba(frame)[:, 1]

    return [
        Prediction(
            probability=float(p),
            risk_band=risk_band(p, artifact.threshold, artifact.medium_threshold),
            threshold=artifact.threshold,
            model_version=artifact.model_version,
        )
        for p in probabilities
    ]
