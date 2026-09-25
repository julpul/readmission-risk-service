"""Train the baseline model on the real dataset and save a servable artifact.

    uv run python -m readmission.models.build_artifact
    uv run python -m readmission.models.build_artifact --data data/raw/diabetic_data.csv \
        --out artifacts/model.joblib

Reuses the experiment code from train.py as-is: thresholds are chosen on
out-of-fold training predictions, metrics are computed once on the held-out test set.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from sklearn.pipeline import Pipeline

from readmission.data.cohort import build_cohort
from readmission.data.loader import load_raw
from readmission.data.schema import validate_raw
from readmission.data.splits import patient_level_split
from readmission.features.pipeline import CATEGORICAL_COLS, NUMERIC_COLS
from readmission.models.artifact import DEFAULT_ARTIFACT_PATH, ModelArtifact, save_artifact
from readmission.models.train import BASELINE_PARAMS, run_experiment, train_model

DEFAULT_DATA_PATH = Path("data") / "raw" / "diabetic_data.csv"
MODEL_NAME = "logreg"
TEST_SIZE = 0.2
MIN_PRECISION = 0.2

# Columns created inside the pipeline - the caller must NOT send these.
DERIVED_COLS = {"age_class", "medication_count", "n_dose_changed", "any_dose_changed"}


def required_input_columns(pipeline: Pipeline) -> list[str]:
    """Raw columns a fitted pipeline reads. Medication columns are learned in fit()."""
    medication_cols = list(pipeline.named_steps["med"].columns_)
    used = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in DERIVED_COLS]
    columns = ["age", *medication_cols, *used]
    return list(dict.fromkeys(columns))  # dedupe, keep order ("insulin" is in both)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_path: Path, n_splits: int = 5) -> ModelArtifact:
    raw = validate_raw(load_raw(data_path))
    cohort = build_cohort(raw)
    data = patient_level_split(cohort, TEST_SIZE)
    X_train, _, y_train, _ = data

    # Two operating points, both picked on OOF predictions (never on test).
    high = run_experiment(
        MODEL_NAME, BASELINE_PARAMS, data, "recall_at_precision", MIN_PRECISION, n_splits
    )
    medium = run_experiment(MODEL_NAME, BASELINE_PARAMS, data, "f1", n_splits=n_splits)

    # Same model + params + data as inside run_experiment -> the same fitted pipeline.
    pipeline = train_model(MODEL_NAME, BASELINE_PARAMS, X_train, y_train)

    trained_at = datetime.now(UTC)
    metric_keys = ["threshold", "pr_auc", "roc_auc", "brier", "precision", "recall", "f1"]
    return ModelArtifact(
        pipeline=pipeline,
        threshold=high["threshold"],
        medium_threshold=min(medium["threshold"], high["threshold"]),
        model_version=f"{MODEL_NAME}-{trained_at:%Y%m%d%H%M%S}",
        trained_at=trained_at,
        input_columns=required_input_columns(pipeline),
        metrics={
            "split": "test",
            "n_test": int(len(data[1])),
            "test_positive_rate": float(data[3].mean()),
            "n_splits": n_splits,
            "recall_at_precision": {k: high[k] for k in [*metric_keys, "n_flagged"]},
            "f1": {k: medium[k] for k in [*metric_keys, "n_flagged"]},
        },
        params=dict(BASELINE_PARAMS),
        data_sha256=file_sha256(data_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    artifact = build(args.data, args.n_splits)
    save_artifact(artifact, args.out)

    # Human-readable sidecar, handy for README/diffs; the .joblib is the source of truth.
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model_version": artifact.model_version,
                "trained_at": artifact.trained_at.isoformat(),
                "params": artifact.params,
                "metrics": artifact.metrics,
            },
            indent=2,
        )
    )
    print(f"saved {args.out} ({artifact.model_version})")
    print(json.dumps(artifact.metrics, indent=2))


if __name__ == "__main__":
    main()
