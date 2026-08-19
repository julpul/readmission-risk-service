import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from readmission.data.constants import AGE_CLASSES, DEAD_OR_HOSPICE


class RawSchema(pa.DataFrameModel):
    encounter_id: Series[int] = pa.Field(unique=True)
    patient_nbr: Series[int]
    race: Series[str] = pa.Field(nullable=True)
    gender: Series[str] = pa.Field(isin=["Male", "Female"], nullable=True)
    age: Series[str] = pa.Field(isin=AGE_CLASSES, nullable=True)
    time_in_hospital: Series[int] = pa.Field(ge=1, le=14)
    discharge_disposition_id: Series[int] = pa.Field(ge=1, le=30)
    number_inpatient: Series[int] = pa.Field(ge=0)
    A1Cresult: Series[str] = pa.Field(isin=["None", "Norm", ">7", ">8"])
    readmitted: Series[str] = pa.Field(isin=["NO", "<30", ">30"])

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True


class CohortSchema(pa.DataFrameModel):
    patient_nbr: Series[int] = pa.Field(unique=True)

    target: Series[int] = pa.Field(isin=[0, 1])
    discharge_disposition_id: Series[int]

    @pa.check("discharge_disposition_id", name="no_death")
    def no_deceased(cls, s: Series[int]):
        return ~s.isin(DEAD_OR_HOSPICE)

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True


def validate_raw(df: pd.DataFrame) -> DataFrame[RawSchema]:
    return RawSchema.validate(df, lazy=True)


def validate_cohort(df: pd.DataFrame) -> pd.DataFrame:
    return CohortSchema.validate(df, lazy=True)
