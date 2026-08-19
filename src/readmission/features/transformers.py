from sklearn.base import BaseEstimator, TransformerMixin

MEDICATION_VALUES = ["No", "Steady", "Up", "Down"]


class AgeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column: str = "age", output_column: str = "age_class", drop_original: bool = True):
        self.column = column
        self.output_column = output_column
        self.drop_original = drop_original

    def fit(self, X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        bounds = X[self.column].str.extract(r"\[(\d+)-(\d+)\)").astype(float)
        X[self.output_column] = bounds.mean(axis=1)
        if self.drop_original:
            X = X.drop(columns=[self.column])
        return X

class MedicationCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, drop_original: bool = False):
        self.columns = columns
        self.drop_original = drop_original

    def fit(self, X, y=None):
        if not self.columns:
            self.columns_ = [col for col in X.columns
                if set(X[col].dropna().unique()) <= set(MEDICATION_VALUES)]
        else:  self.columns_ = list(self.columns)
        return self

    def transform(self, X):
        X = X.copy()
        meds = X[self.columns_]
        X["medication_count"] = (meds != "No").sum(axis=1)
        X["n_dose_changed"] = (meds.isin(["Up", "Down"])).sum(axis=1)
        X["any_dose_changed"] = (X["n_dose_changed"] > 0).astype(int)
        if self.drop_original:
            X = X.drop(columns=self.columns_)
        return X
