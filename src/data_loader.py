"""Load and clean the raw salary CSV.

Cleaning steps (all counted in the returned report):
1. normalise column names; strip whitespace; normalise education / gender labels
2. drop rows with no Salary (cannot be used for supervised training)
3. drop implausible salaries (< MIN_PLAUSIBLE_SALARY)
4. drop exact duplicate rows -- this MUST happen before the train/test split,
   otherwise identical rows land on both sides and inflate the test metrics
Rows with missing *feature* values are kept and imputed inside the pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH,
    EDUCATION_MAP,
    GENDER_VALUES,
    MIN_PLAUSIBLE_SALARY,
    RAW_FEATURES,
    TARGET_COLUMN,
)


def normalize_column_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def normalize_education(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return np.nan
    return EDUCATION_MAP.get(str(value).strip().lower(), np.nan)


def normalize_gender(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().capitalize()
    return text if text in GENDER_VALUES else np.nan


def normalize_title(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or np.nan


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    required = RAW_FEATURES + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {list(df.columns)}")
    df = df[required]
    report = {"rows_raw": int(len(df))}

    unmapped_edu = int(
        (df["Education_Level"].notna() & df["Education_Level"].map(normalize_education).isna()).sum()
    )
    df["Education_Level"] = df["Education_Level"].map(normalize_education)
    df["Gender"] = df["Gender"].map(normalize_gender)
    df["Job_Title"] = df["Job_Title"].map(normalize_title)
    for col in ("Age", "Years_of_Experience", TARGET_COLUMN):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # object dtype keeps missing values as plain NaN for SimpleImputer
    for col in ("Gender", "Education_Level", "Job_Title"):
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    report["unrecognised_education_labels_set_to_missing"] = unmapped_edu

    n = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    report["dropped_missing_salary"] = int(n - len(df))

    n = len(df)
    df = df[df[TARGET_COLUMN] >= MIN_PLAUSIBLE_SALARY]
    report["dropped_implausible_salary"] = int(n - len(df))

    n = len(df)
    df = df.drop_duplicates()
    report["dropped_exact_duplicates"] = int(n - len(df))

    df = df.reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["rows_with_missing_features_kept_for_imputation"] = int(
        df[RAW_FEATURES].isna().any(axis=1).sum()
    )
    return df, report


def load_data(path: Path | str | None = None, with_report: bool = False):
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See data/README.md.")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    df, report = clean_data(raw)
    return (df, report) if with_report else df
