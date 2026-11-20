"""
data_prep.py - loads and cleans waterQuality1.csv.

Every later step (training, API, dashboard) imports load_clean() from here,
so the cleaning rules live in exactly ONE place.
"""
from pathlib import Path

import pandas as pd

# Project root = the folder that contains src/, data/, artifacts/ ...
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "waterQuality1.csv"
TARGET = "is_safe"


def load_clean(path=DATA_PATH, verbose=True):
    """Return (clean_dataframe, report_dict)."""
    # na_values turns the Excel error text '#NUM!' into a proper missing value (NaN).
    # Without this, pandas would read 'ammonia' and 'is_safe' as text columns.
    df = pd.read_csv(path, na_values=["#NUM!"])
    n_raw = len(df)

    # Fix the typo in the column name (flouride -> fluoride).
    # From now on use "fluoride" everywhere (limits.json, API, frontend).
    df = df.rename(columns={"flouride": "fluoride"})

    # Rule 1: drop rows with missing values (the 3 '#NUM!' rows).
    n_missing = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    # Rule 2: a concentration cannot be negative -> sensor/entry error, drop.
    n_negative = int((df["ammonia"] < 0).sum())
    df = df[df["ammonia"] >= 0]

    # The target must be a clean integer 0/1.
    df[TARGET] = df[TARGET].astype(int)
    df = df.reset_index(drop=True)

    report = {
        "raw_rows": n_raw,
        "dropped_missing": n_missing,
        "dropped_negative_ammonia": n_negative,
        "clean_rows": len(df),
        "safe_rate": round(float(df[TARGET].mean()), 4),
    }
    if verbose:
        print("Cleaning report:", report)
    return df, report


def feature_columns(df):
    """All columns except the label."""
    return [c for c in df.columns if c != TARGET]


if __name__ == "__main__":
    load_clean()
