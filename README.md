# Readmission Risk Service

Predicts the risk that a diabetic inpatient is **readmitted to hospital within 30 days of
discharge**, and serves the model behind a small, tested FastAPI service.

> **Not a clinical tool.** This is a learning / portfolio project built on a public,
> de-identified research dataset from 1999–2008. Do not use it to make decisions about real
> patients.

---

## Why this problem

Unplanned 30-day readmissions are expensive and often a sign that something went wrong at
or after discharge (medication changes, missed follow-up, an unresolved condition). In the
US, the CMS **Hospital Readmissions Reduction Program** financially penalises hospitals with
excess 30-day readmissions, so hospitals want to know *which patients to follow up with*
before they leave.

The realistic product is therefore not "a diagnosis" but a **ranking plus a cut-off**:
a care-coordination team can call a limited number of patients per day, so the model has to
put the riskiest patients at the top and the threshold has to respect team capacity.

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008** — UCI Machine Learning Repository
([dataset page](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)),
introduced by Strack et al., *"Impact of HbA1c Measurement on Hospital Readmission Rates"*,
BioMed Research International, 2014.

| | |
|---|---|
| encounters | 101,766 |
| unique patients | 71,518 |
| columns | 50 (demographics, admission info, ICD-9 diagnoses, 23 diabetes drugs, labs) |
| label | `readmitted` ∈ {`<30`, `>30`, `NO`} |

The raw CSV is not committed. Download the zip from UCI and place
`diabetic_data.csv` (and optionally `IDS_mapping.csv`) in `data/raw/`.

## Approach

```
diabetic_data.csv                 101,766 rows x 50 cols
   │ load_raw           "?" -> NA, but "None" in A1Cresult stays a category ("test not ordered")
   │ RawSchema          pandera: types, ranges, allowed values
   │ build_cohort       -> 69,990 rows, target rate 8.98%
   │ patient_level_split  80/20 by patient -> 55,992 train / 13,998 test
   │ feature pipeline   (sklearn Pipeline, fitted on train only)
   ▼
LogisticRegression -> score -> threshold chosen on out-of-fold train predictions
```

**Cohort definition** (`src/readmission/data/cohort.py`)

- **Target** = readmitted in `<30` days; `>30` is treated as negative (the 30-day window is
  what hospitals are measured on).
- **Deceased and hospice discharges are removed** (disposition codes 11, 13, 14, 19, 20, 21).
  They are negatives only because they *could not* come back; keeping them teaches the model
  to predict mortality. This drops the positive rate from 11.16% to 8.98%.
- **One encounter per patient** (the earliest). Rows of the same patient are not independent.

**Leakage control**

- Train/test split and cross-validation are **grouped by `patient_nbr`**
  (`GroupShuffleSplit`, `StratifiedGroupKFold`), so no patient appears on both sides.
  (With the current one-encounter-per-patient cohort the groups are trivially singletons; the
  grouping matters once all encounters are used.)
- Everything that learns from data (imputation, scaling, one-hot encoding, medication-column
  detection) lives **inside the sklearn `Pipeline`**, so it is fitted on training folds only.
- `ColumnTransformer(remainder="drop")` acts as an allow-list — identifiers
  (`encounter_id`, `patient_nbr`) and the label never reach the model. An API test asserts that
  they do not appear among the model features.
- The decision threshold is picked on **out-of-fold** training predictions; the test set is
  touched exactly once, for the final metrics.

**Schema validation** — [pandera](https://pandera.readthedocs.io/) `DataFrameModel`s validate
the raw file (`RawSchema`) and the cohort (`CohortSchema`, e.g. "no deceased patients left").
At serving time, a pydantic model does the same job for each request.

**Features** (`src/readmission/features/`)

- **ICD-9 grouping** — ~900 distinct ICD-9 codes across `diag_1..3` mapped to 9 groups
  (circulatory, respiratory, digestive, diabetes, injury, …) following Strack et al.
- **Drug aggregation** — 23 sparse medication columns collapsed into `medication_count` and
  `n_dose_changed` instead of 92 mostly-empty one-hot columns.
- **Age** — `[70-80)` brackets converted to the numeric midpoint.
- Numeric features are median-imputed and standardised; categoricals are one-hot encoded with
  `handle_unknown="ignore"` (119 model features in total).

## Model & metrics

Baseline: `LogisticRegression(class_weight="balanced", max_iter=1000)`.
Produced by `uv run python -m readmission.models.build_artifact` (5-fold grouped CV for the
threshold, metrics on the held-out test set of 13,998 patients, positive rate 9.02%).

| metric (test set) | value | note |
|---|---:|---|
| ROC-AUC | **0.607** | |
| PR-AUC (average precision) | **0.139** | random = 0.090 → 1.54× better than random |
| Brier score | 0.238 | poor — scores are *not* calibrated probabilities (see limitations) |

Two operating points, both chosen on out-of-fold training data:

| operating point | threshold | precision | recall | flagged on test |
|---|---:|---:|---:|---:|
| `recall_at_precision` (target precision ≥ 0.20) → band **high** | 0.665 | 0.195 | 0.090 | 585 (4.2%) |
| `f1`-optimal → band **medium** and above | 0.499 | 0.122 | 0.539 | 5,593 (40.0%) |

Honest reading: the model ranks patients better than chance, but only modestly. The
capacity-friendly threshold misses its precision target slightly on unseen data (0.195 vs
0.20), which is expected for a point chosen exactly on the boundary. More training data does
not help (tested); the bottleneck is the feature set — see the roadmap.

## API

Start the server (requires a trained artifact, see [How to run](#how-to-run)):

```bash
uv run uvicorn readmission.api.app:app --reload
# interactive docs: http://127.0.0.1:8000/docs
```

| endpoint | description |
|---|---|
| `GET /health` | liveness; always 200, reports `model_loaded` |
| `GET /model/info` | version, `trained_at`, thresholds, params, test metrics, input columns, model features |
| `POST /predict` | score one encounter |
| `POST /predict/batch` | score up to 1,000 encounters, order preserved |

The model is loaded once at startup. If the artifact is missing or incompatible, the
service still starts, `/health` reports `"model_loaded": false`, and model endpoints return
**503** with the reason. Invalid requests return **422**.

The request body mirrors the raw dataset columns — the service does the feature engineering.
Medication fields default to `"No"`; unknown fields are rejected (a typo like `insullin`
must not silently become "not prescribed").

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "age": "[70-80)", "race": "Caucasian", "gender": "Female",
        "time_in_hospital": 5, "num_lab_procedures": 45,
        "number_inpatient": 2, "number_emergency": 0,
        "medical_specialty": "InternalMedicine",
        "diag_1": "428", "diag_2": "250.02", "diag_3": "401",
        "change": "Ch", "insulin": "Up", "metformin": "Steady"
      }'
# {"probability":0.7375,"risk_band":"high","threshold":0.6652,"model_version":"logreg-20260925123222"}

curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"encounters": [ {...}, {...} ]}'
# {"predictions":[{...},{...}]}
```

`risk_band`: `high` if score ≥ `threshold`, `medium` if score ≥ the F1 operating point,
otherwise `low`.

## Project structure

```
.
├── src/readmission/
│   ├── data/          # loading, pandera schemas, cohort rules, patient-level splits
│   ├── features/      # ICD-9 grouping, medication/age transformers, sklearn pipeline
│   ├── models/        # training, threshold selection & metrics, artifact save/load
│   │   └── build_artifact.py   # CLI: train on the real data -> artifacts/model.joblib
│   └── api/           # FastAPI app, pydantic schemas, prediction glue
├── tests/             # pytest: data, features, api (api tests need no dataset)
├── notebooks/         # EDA
├── docs/              # working notes (PL)
├── data/raw/          # dataset goes here (not committed)
├── artifacts/         # trained model (not committed)
├── Dockerfile
└── .github/workflows/ci.yml
```

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync                                              # install deps (incl. dev group)

uv run python -m readmission.models.build_artifact   # train -> artifacts/model.joblib
                                                     #   (+ artifacts/model.metrics.json)
uv run uvicorn readmission.api.app:app --reload      # serve on :8000
uv run pytest                                        # run tests
uv run ruff check . && uv run ruff format --check .  # lint
```

Use another artifact location with `MODEL_PATH=/path/to/model.joblib`.

**Docker** — the image contains code and dependencies only; the model is mounted at runtime:

```bash
docker build -t readmission-risk-service .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" readmission-risk-service
```

The container runs as a non-root user and has a `HEALTHCHECK` on `/health`.

## Testing & CI

- **Unit tests** for cohort rules (deaths/hospice removed, one encounter per patient,
  target mapping), the patient-level split (no shared patients), and ICD-9 mapping edge
  cases (`250.83`, `V45`, missing).
- **API tests** (`tests/api/`) use FastAPI's `TestClient` against a tiny pipeline fitted on
  synthetic rows — the *real* feature pipeline, only the training data is fake — so they run
  in about a second without the dataset. They cover the response contract, 503 without a
  model, 422 on invalid/unknown/missing fields, batch limits and ordering, risk-band
  boundaries, and that the API returns exactly the pipeline's score.
- **CI** (GitHub Actions): `uv sync --frozen`, `ruff check`, `ruff format --check`, `pytest`
  on every push to `main` and every pull request.

## Limitations & ethics

- **Not a clinical tool**, not validated prospectively, not reviewed by clinicians.
- **Scores are not calibrated probabilities.** `class_weight="balanced"` inflates them
  (mean predicted score ≈ 0.48 vs an actual rate of 0.09). Treat `probability` as a ranking
  score; the Brier score is worse than a constant predictor (≈ 0.082).
- **Old and US-only data** (1999–2008, 130 US hospitals). Care practices, coding (ICD-9 →
  ICD-10) and drugs have changed since.
- **Dataset bias.** `race` is missing for ~2% of encounters and group sizes are very
  uneven; `weight` is missing for 97% and `payer_code` for 40%, so neither is used. Performance per subgroup has not been audited yet, and a
  model like this can encode differences in access to care rather than clinical risk.
- Readmissions to *other* hospitals are not observed in the data.
- Only the first encounter per patient is used, which removes most patients with prior
  admissions from training.

## Roadmap

Done:

- [x] Data loading with pandera validation, cohort definition, patient-level splits
- [x] Feature pipeline (ICD-9 groups, medication aggregation, age)
- [x] Logistic-regression baseline, OOF threshold selection, test-set metrics
- [x] Model artifact with version, thresholds and metrics
- [x] FastAPI service with tests, Dockerfile, CI (lint + tests)

Planned:

- [ ] Add missing features with known signal (`number_diagnoses`, `number_outpatient`,
      `A1Cresult`, `admission_type_id`, `discharge_disposition_id`, …)
- [ ] Collapse rare `medical_specialty` values (top-N learned in `fit`)
- [ ] Drop `class_weight="balanced"` and/or calibrate, so scores mean something
- [ ] Safety margin when picking the precision-constrained threshold
- [ ] Gradient boosting and a proper experiment table
- [ ] Subgroup performance report (race, gender, age)
- [ ] Tests for `evaluate.py` and a split test on a multi-encounter cohort
- [ ] Build the Docker image in CI; request logging and basic drift monitoring
