from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def patient_level_split(df, test_size, random_state=42):
    ggs = GroupShuffleSplit(n_splits=1, random_state=random_state, test_size=test_size)

    train_idx, test_idx = next(ggs.split(df, groups=df["patient_nbr"]))
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    X_train = df_train.drop(columns="target")
    y_train = df_train["target"]

    X_test = df_test.drop(columns="target")
    y_test = df_test["target"]

    return (X_train, X_test, y_train, y_test)


def make_cv_splitter(n_splits):
    return StratifiedGroupKFold(n_splits=n_splits)
