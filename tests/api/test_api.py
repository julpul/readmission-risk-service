import pandas as pd
import pytest

from readmission.api.predict import risk_band
from readmission.api.schemas import MAX_BATCH_SIZE, Encounter


def predict_directly(artifact, body: dict) -> float:
    """Score a request body with the pipeline itself, bypassing the API."""
    row = Encounter.model_validate(body).to_row()
    return float(artifact.pipeline.predict_proba(pd.DataFrame([row]))[0, 1])


# --------------------------------------------------------------------- health
def test_health_with_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_without_model_is_still_ok(client_without_model):
    # liveness must not depend on the model - otherwise the container restarts forever
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


@pytest.mark.parametrize(
    "method, url", [("get", "/model/info"), ("post", "/predict"), ("post", "/predict/batch")]
)
def test_model_endpoints_return_503_without_model(
    client_without_model, valid_encounter, method, url
):
    body = {"encounters": [valid_encounter]} if url.endswith("batch") else valid_encounter
    kwargs = {"json": body} if method == "post" else {}

    response = getattr(client_without_model, method)(url, **kwargs)

    assert response.status_code == 503
    assert "model not loaded" in response.json()["detail"]


# ----------------------------------------------------------------- model info
def test_model_info(client, dummy_artifact):
    info = client.get("/model/info").json()

    assert info["model_version"] == "test-0"
    assert info["threshold"] == dummy_artifact.threshold
    assert info["metrics"] == {"pr_auc": 0.5}
    assert "age" in info["input_columns"]
    assert "glyburide-metformin" in info["input_columns"]
    assert info["model_features"]  # feature names after preprocessing


def test_model_info_does_not_leak_identifiers(client):
    features = client.get("/model/info").json()["model_features"]
    for forbidden in ["encounter_id", "patient_nbr", "readmitted", "target"]:
        assert not any(forbidden in f for f in features)


# -------------------------------------------------------------------- predict
def test_predict_returns_contract(client, valid_encounter):
    response = client.post("/predict", json=valid_encounter)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probability", "risk_band", "threshold", "model_version"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"] == "test-0"


def test_predict_matches_pipeline(client, dummy_artifact, valid_encounter):
    # The API must not change the number: same input -> same score as the raw pipeline.
    api_proba = client.post("/predict", json=valid_encounter).json()["probability"]
    assert api_proba == pytest.approx(predict_directly(dummy_artifact, valid_encounter))


def test_band_is_consistent_with_threshold(client, valid_encounter):
    body = client.post("/predict", json=valid_encounter).json()
    expected = risk_band(body["probability"], body["threshold"], 0.4)
    assert body["risk_band"] == expected


def test_predict_accepts_missing_optional_fields(client, valid_encounter):
    for key in ["race", "gender", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        valid_encounter[key] = None

    assert client.post("/predict", json=valid_encounter).status_code == 200


def test_predict_accepts_hyphenated_medication(client, valid_encounter):
    valid_encounter["glyburide-metformin"] = "Steady"
    assert client.post("/predict", json=valid_encounter).status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", "[70-79)"),  # not a dataset bracket
        ("age", "75"),
        ("time_in_hospital", 0),  # dataset range is 1..14
        ("time_in_hospital", 15),
        ("number_inpatient", -1),
        ("insulin", "Maybe"),
        ("gender", "Unknown/Invalid"),
        ("change", "Yes"),
    ],
)
def test_predict_rejects_invalid_values(client, valid_encounter, field, value):
    valid_encounter[field] = value
    assert client.post("/predict", json=valid_encounter).status_code == 422


def test_predict_rejects_unknown_field(client, valid_encounter):
    valid_encounter["insullin"] = "Up"  # typo must not silently become insulin="No"
    assert client.post("/predict", json=valid_encounter).status_code == 422


@pytest.mark.parametrize("field", ["age", "time_in_hospital", "number_inpatient"])
def test_predict_rejects_missing_required_field(client, valid_encounter, field):
    del valid_encounter[field]
    assert client.post("/predict", json=valid_encounter).status_code == 422


# ---------------------------------------------------------------------- batch
def test_batch_preserves_order_and_matches_single(client, valid_encounter):
    low_risk = {**valid_encounter, "number_inpatient": 0}
    high_risk = {**valid_encounter, "number_inpatient": 3}

    batch = client.post("/predict/batch", json={"encounters": [low_risk, high_risk]})
    assert batch.status_code == 200
    predictions = batch.json()["predictions"]

    assert len(predictions) == 2
    for sent, got in zip([low_risk, high_risk], predictions, strict=True):
        single = client.post("/predict", json=sent).json()
        assert got["probability"] == pytest.approx(single["probability"])


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_limits(client, valid_encounter, size):
    response = client.post("/predict/batch", json={"encounters": [valid_encounter] * size})
    assert response.status_code == 422


# ----------------------------------------------------------------- risk bands
@pytest.mark.parametrize(
    "probability, band",
    [
        (0.0, "low"),
        (0.39, "low"),
        (0.4, "medium"),  # boundary belongs to the upper band (>=)
        (0.59, "medium"),
        (0.6, "high"),
        (1.0, "high"),
    ],
)
def test_risk_band_boundaries(probability, band):
    assert risk_band(probability, threshold=0.6, medium_threshold=0.4) == band


def test_schema_covers_every_raw_column_the_model_needs(dummy_artifact):
    assert set(dummy_artifact.input_columns) <= Encounter.raw_columns()
