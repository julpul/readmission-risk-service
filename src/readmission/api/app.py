"""FastAPI service.

    uv run uvicorn readmission.api.app:app --reload

The model is loaded once at startup (lifespan). If the artifact is missing or
broken the service still starts - /health reports model_loaded=false and the
model endpoints answer 503, so the problem is visible instead of a crash loop.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from readmission.api.predict import missing_columns, predict
from readmission.api.schemas import (
    BatchRequest,
    BatchResponse,
    Encounter,
    Health,
    ModelInfo,
    Prediction,
)
from readmission.models.artifact import DEFAULT_ARTIFACT_PATH, ModelArtifact, load_artifact

logger = logging.getLogger(__name__)

MODEL_PATH_ENV = "MODEL_PATH"


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path
    return Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_ARTIFACT_PATH))


def try_load(path: Path) -> tuple[ModelArtifact | None, str | None]:
    """Return (artifact, None) on success or (None, reason) on failure."""
    try:
        artifact = load_artifact(path)
    except Exception as exc:  # any load failure -> 503, never a crash at startup
        logger.error("could not load model from %s: %s", path, exc)
        return None, f"{type(exc).__name__}: {exc}"

    missing = missing_columns(artifact)
    if missing:
        reason = f"model needs columns the API schema does not accept: {missing}"
        logger.error(reason)
        return None, reason

    logger.info("loaded model %s from %s", artifact.model_version, path)
    return artifact, None


def create_app(model_path: Path | None = None) -> FastAPI:
    path = resolve_model_path(model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.artifact, app.state.load_error = try_load(path)
        yield

    app = FastAPI(
        title="Readmission Risk Service",
        description="30-day readmission risk for diabetic inpatients. Not a clinical tool.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_artifact(request: Request) -> ModelArtifact:
        artifact = request.app.state.artifact
        if artifact is None:
            raise HTTPException(
                status_code=503,
                detail=f"model not loaded ({request.app.state.load_error})",
            )
        return artifact

    Artifact = Annotated[ModelArtifact, Depends(get_artifact)]

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model_loaded=request.app.state.artifact is not None)

    @app.get("/model/info")
    def model_info(artifact: Artifact) -> ModelInfo:
        features = artifact.pipeline.named_steps["pre"].get_feature_names_out()
        return ModelInfo(
            model_version=artifact.model_version,
            trained_at=artifact.trained_at,
            sklearn_version=artifact.sklearn_version,
            threshold=artifact.threshold,
            medium_threshold=artifact.medium_threshold,
            params=artifact.params,
            metrics=artifact.metrics,
            input_columns=artifact.input_columns,
            model_features=[str(f) for f in features],
            data_sha256=artifact.data_sha256,
        )

    @app.post("/predict")
    def predict_one(encounter: Encounter, artifact: Artifact) -> Prediction:
        return predict(artifact, [encounter])[0]

    @app.post("/predict/batch")
    def predict_batch(batch: BatchRequest, artifact: Artifact) -> BatchResponse:
        return BatchResponse(predictions=predict(artifact, batch.encounters))

    return app


app = create_app()
