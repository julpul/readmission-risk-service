from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .icd9 import Icd9GroupTransformer
from .transformers import AgeTransformer, MedicationCountTransformer

NUMERIC_COLS = [
    "age_class",
    "time_in_hospital",
    "medication_count",
    "n_dose_changed",
    "num_lab_procedures",
    "number_inpatient",
    "number_emergency",
]
CATEGORICAL_COLS = [
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "race",
    "gender",
    "insulin",
    "change",
]


def build_preprocessor(numeric_calls, categorical_colls) -> ColumnTransformer:
    transformers = [
        (
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            numeric_calls,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            categorical_colls,
        ),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_full_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("age", AgeTransformer()),
            ("med", MedicationCountTransformer()),
            ("icd9", Icd9GroupTransformer()),
            ("pre", build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS)),
            ("clf", model),
        ]
    )
