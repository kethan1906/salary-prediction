"""Derived features, applied INSIDE the saved pipeline (FunctionTransformer) so
training and prediction can never disagree about them."""
import re

import numpy as np
import pandas as pd

CAREER_BINS = [-1, 2, 7, 15, float("inf")]
CAREER_LABELS = ["Early_Career", "Mid_Career", "Experienced", "Senior"]

_EXEC = re.compile(r"\b(chief|ceo|cto|cfo|coo|cmo|vp|vice president)\b")
_SENIOR = re.compile(r"\b(senior|sr\.?|lead|principal)\b")
_JUNIOR = re.compile(r"\b(junior|jr\.?|intern|entry|trainee)\b")


def title_level(title) -> object:
    """Seniority bucket from job-title keywords. Generalises to unseen titles."""
    if not isinstance(title, str):
        return np.nan
    t = title.lower()
    if _EXEC.search(t):
        return "Executive"
    if "director" in t:
        return "Director"
    if "manager" in t or "head of" in t:
        return "Manager"
    if _SENIOR.search(t):
        return "Senior"
    if _JUNIOR.search(t):
        return "Junior"
    return "Standard"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Experience_Squared, Career_Stage and Title_Level (returns a copy)."""
    df = df.copy()
    if "Years_of_Experience" in df.columns:
        years = pd.to_numeric(df["Years_of_Experience"], errors="coerce")
        df["Years_of_Experience"] = years
        df["Experience_Squared"] = years ** 2
        df["Career_Stage"] = pd.cut(years, bins=CAREER_BINS, labels=CAREER_LABELS).astype(object)
    if "Job_Title" in df.columns:
        df["Title_Level"] = df["Job_Title"].map(title_level).astype(object)
    return df
