import pandas as pd
import pytest

from readmission.data import patient_level_split

PATIENTS = {
    100: (3, 1),
    101: (3, 0),
    102: (3, 0),
    103: (3, 0),
    104: (3, 0),
    105: (2, 1),
    106: (2, 0),
    107: (2, 0),
    108: (2, 0),
    109: (2, 0),
    110: (1, 0),
    111: (1, 0),
    112: (1, 1),
    113: (1, 0),
    114: (1, 0),
    115: (1, 0),
    116: (1, 0),
    117: (1, 1),
    118: (1, 0),
    119: (1, 0),
}


@pytest.fixture
def cohort_df():
    rows = []
    encounter_id = 1
    for patient_nbr, (n_visits, target) in PATIENTS.items():
        for _ in range(n_visits):
            rows.append(
                {
                    "encounter_id": encounter_id,
                    "patient_nbr": patient_nbr,
                    "time_in_hospital": (encounter_id % 14) + 1,
                    "number_inpatient": encounter_id % 5,
                    "target": target,
                }
            )
            encounter_id += 1
    return pd.DataFrame(rows)


def test_fixture_shape(cohort_df):
    assert len(cohort_df) == 35
    assert cohort_df["patient_nbr"].nunique() == 20
    assert cohort_df["patient_nbr"].duplicated().any()  # patients with >1 visit exist
    assert cohort_df["target"].mean() == pytest.approx(0.20)


# TODO - to be written:
def test_no_shared_patients(cohort_df):
    X_train, X_test, y_train, y_test = patient_level_split(cohort_df, 0.2)
    assert set(X_train["patient_nbr"]) & set(X_test["patient_nbr"]) == set()
    assert X_train["patient_nbr"].isin(X_test["patient_nbr"]).sum() == 0


#
# 2. test_class_proportion_preserved
#    positive class ratio similar in train and test.
#    Hint: pytest.approx(..., abs=0.15) - with 35 rows the tolerance must be
#    loose, otherwise the test will flake


#
# 3. test_no_rows_lost
#    len(train) + len(test) == len(cohort_df)
#
# 4. test_reproducible
#    two calls with the same random_state produce an identical split
#
# 5. test_cv_splitter_keeps_patients_together
#    for every fold from make_cv_splitter the patient intersection is empty.
#    Hint: for train_idx, test_idx in cv.split(X, y, groups=...)
