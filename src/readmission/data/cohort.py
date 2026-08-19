import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from readmission.data.constants import DEAD_OR_HOSPICE
from readmission.data.schema import CohortSchema, RawSchema


def drop_deceased_and_hospice(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df["discharge_disposition_id"].isin(DEAD_OR_HOSPICE)].copy()


def duplicate_patients(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "first":
        return df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    if strategy == "all":
        return df
    raise ValueError(f"unknown strategy: {strategy!r}")


def make_target(df: pd.DataFrame) -> pd.Series:
    return (df["readmitted"] == "<30").astype(int)


@pa.check_types
def build_cohort(df: DataFrame[RawSchema]) -> DataFrame[CohortSchema]:
    out = drop_deceased_and_hospice(df)
    out = duplicate_patients(out, "first")
    out = out.assign(target=make_target(out))
    return DataFrame[CohortSchema](out)
