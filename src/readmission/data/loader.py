from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, na_values=["?"])
    return frame.replace("Unknown/Invalid", pd.NA)
