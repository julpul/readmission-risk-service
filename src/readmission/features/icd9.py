import pandas as pd

ICD9_GROUPS = {
    "circulatory":     [(390, 459), (785, 785)],
    "respiratory":     [(460, 519), (786, 786)],
    "digestive":       [(520, 579), (787, 787)],
    "genitourinary":   [(580, 629), (788, 788)],
    "neoplasms":       [(140, 239)],
    "musculoskeletal": [(710, 739)],
    "injury":          [(800, 999)],
}
DIABETES_PREFIX = "250"


def map_icd9_to_group(code: str | float | None) -> str:
    if code is None or pd.isna(code) or code:
        return "other"

    text = str(code).strip()
    if not text:
        return "other"

    if text.startswith(("V", "E", "v", "e")):
        return "other"

    if text.startswith(DIABETES_PREFIX):
        return "diabetes"

    try:
        code_int = int(float(text))
    except ValueError:
        return "other"

    for group, ranges in ICD9_GROUPS.items():
        for start, end in ranges:
            if start <= code_int <= end:
                return group

    return "other"

class Icd9GroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = ("diag_1","diag_2","diag_3"), drop_original: bool = True):
        self.columns = columns
        self.drop_original = drop_original

    def fit(self,X,y):
        return self

    def transform(self,X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(map_icd9_to_group)
        return X

 