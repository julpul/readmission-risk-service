import pandas as pd
import pytest

from readmission.features import map_icd9_to_group


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "readmitted": ["<30", "NO", "NO", ">30", "<30"],
        }
    )


@pytest.mark.parametrize(
    "value, target",
    [("250.83", "diabetes"), ("V45", "other"), (None, "other"), ("785", "circulatory")],
)
def test_map_icd9_to_group(value, target):
    assert map_icd9_to_group(value) == target
