"""Request/response contracts of the API.

`Encounter` mirrors the raw CSV columns the pipeline reads - the service does the
feature engineering itself, so clients send data in the same shape as the dataset.
Field names with hyphens (e.g. "glyburide-metformin") are exposed via aliases.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from readmission.data.constants import AGE_CLASSES

Medication = Literal["No", "Steady", "Up", "Down"]
RiskBand = Literal["low", "medium", "high"]

MAX_BATCH_SIZE = 1000


class Encounter(BaseModel):
    # forbid: a typo like "insullin" must fail loudly, not silently default to "No"
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    # demographics
    age: str = Field(description="Age bracket as in the dataset, e.g. '[70-80)'")
    race: str | None = None
    gender: Literal["Male", "Female"] | None = None

    # admission
    time_in_hospital: int = Field(ge=1, le=14, description="Length of stay in days")
    num_lab_procedures: int = Field(ge=0)
    number_inpatient: int = Field(ge=0, description="Inpatient visits in the prior year")
    number_emergency: int = Field(ge=0, description="Emergency visits in the prior year")
    medical_specialty: str | None = None

    # ICD-9 codes as strings, e.g. "250.83", "V45", "428"
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None

    change: Literal["No", "Ch"] = Field("No", description="Any diabetes medication changed")

    # 23 medication columns; default "No" = not prescribed
    metformin: Medication = "No"
    repaglinide: Medication = "No"
    nateglinide: Medication = "No"
    chlorpropamide: Medication = "No"
    glimepiride: Medication = "No"
    acetohexamide: Medication = "No"
    glipizide: Medication = "No"
    glyburide: Medication = "No"
    tolbutamide: Medication = "No"
    pioglitazone: Medication = "No"
    rosiglitazone: Medication = "No"
    acarbose: Medication = "No"
    miglitol: Medication = "No"
    troglitazone: Medication = "No"
    tolazamide: Medication = "No"
    examide: Medication = "No"
    citoglipton: Medication = "No"
    insulin: Medication = "No"
    glyburide_metformin: Medication = Field("No", alias="glyburide-metformin")
    glipizide_metformin: Medication = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Medication = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Medication = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Medication = Field("No", alias="metformin-pioglitazone")

    @field_validator("age")
    @classmethod
    def age_in_known_classes(cls, value: str) -> str:
        if value not in AGE_CLASSES:
            raise ValueError(f"age must be one of {AGE_CLASSES}")
        return value

    def to_row(self) -> dict:
        """Dict keyed by raw dataset column names (hyphenated aliases)."""
        return self.model_dump(by_alias=True)

    @classmethod
    def raw_columns(cls) -> set[str]:
        return {field.alias or name for name, field in cls.model_fields.items()}


class BatchRequest(BaseModel):
    encounters: list[Encounter] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    probability: float = Field(description="Model score for 30-day readmission (see README)")
    risk_band: RiskBand
    threshold: float = Field(description="Score at or above which the band is 'high'")
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfo(BaseModel):
    model_version: str
    trained_at: datetime
    sklearn_version: str
    threshold: float
    medium_threshold: float
    params: dict
    metrics: dict
    input_columns: list[str]
    model_features: list[str]
    data_sha256: str | None
