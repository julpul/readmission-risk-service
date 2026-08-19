import pandas as pd
import pytest

from readmission.data import drop_deceased_and_hospice, duplicate_patients, make_target


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "patient_nbr": [100, 100, 200, 300, 400],
            "age": ["[70-80)"] * 5,
            "time_in_hospital": [3, 5, 2, 1, 8],
            "discharge_disposition_id": [1, 1, 11, 6, 13],
            "readmitted": ["<30", "NO", "NO", ">30", "<30"],
        }
    )


def test_deaths_out(raw_df):
    out = drop_deceased_and_hospice(raw_df)
    assert out["discharge_disposition_id"].tolist() == [1, 1, 6]


def test_duplicates(raw_df):
    out = duplicate_patients(raw_df, "first")
    assert len(out) == 4
    assert out.loc[out["patient_nbr"] == 100, "encounter_id"].item() == 1


def test_duplicates_all(raw_df):
    assert len(duplicate_patients(raw_df, "all")) == 5


@pytest.mark.parametrize("value, target", [("<30", 1), (">30", 0), ("NO", 0)])
def test_target_make(raw_df, value, target):
    df = raw_df.assign(readmitted=value)
    assert make_target(df).unique().tolist() == [target]
