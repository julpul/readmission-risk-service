"""Model training and experiment runner."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from readmission.data.splits import make_cv_splitter
from readmission.features.pipeline import build_full_pipeline
from readmission.models.evaluate import compute_metrics, find_threshold

MODELS = {
    "logreg": LogisticRegression,
}

BASELINE_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
}


def train_model(model_name: str, params: dict, X_train, y_train) -> Pipeline:
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name!r}; available: {sorted(MODELS)}")

    model = MODELS[model_name](**params)
    pipeline = build_full_pipeline(model)
    return pipeline.fit(X_train, y_train)


def train_baseline(X_train, y_train) -> Pipeline:
    return train_model("logreg", BASELINE_PARAMS, X_train, y_train)


def run_experiment(
    model_name: str,
    params: dict,
    data: tuple,
    criterion: str = "recall_at_precision",
    min_precision: float = 0.2,
    n_splits: int = 5,
) -> dict:
    X_train, X_test, y_train, y_test = data
    cv = make_cv_splitter(n_splits)
    oof_proba = cross_val_predict(
        build_full_pipeline(MODELS[model_name](**params)),
        X_train,
        y_train,
        cv=cv,
        groups=X_train["patient_nbr"],
        method="predict_proba",
    )[:, 1]

    threshold = find_threshold(y_train, oof_proba, criterion, min_precision)

    pipeline = train_model(model_name, params, X_train, y_train)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_proba, threshold)

    return {
        "model": model_name,
        "criterion": criterion,
        **params,
        **metrics,
    }
