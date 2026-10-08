"""Column roles and the ColumnTransformer shared by every candidate model."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    EDUCATION_ORDER,
    HIGH_CARDINALITY_FEATURES,
    JOB_TITLE_MIN_FREQUENCY,
    LOW_CARDINALITY_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)


def select_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """Feature roles restricted to columns present in ``df`` (post feature engineering)."""
    pick = lambda cols: [c for c in cols if c in df.columns]  # noqa: E731
    return {
        "numeric": pick(NUMERIC_FEATURES),
        "ordinal": pick(ORDINAL_FEATURES),
        "low_cardinality": pick(LOW_CARDINALITY_FEATURES),
        "high_cardinality": pick(HIGH_CARDINALITY_FEATURES),
    }


def build_preprocessor(roles: dict[str, list[str]]) -> ColumnTransformer:
    """Fresh, unfitted preprocessor.

    numeric          -> median impute -> standard scale
    ordinal          -> mode impute  -> ordered integer (High School < ... < PhD) -> scale
    low_cardinality  -> mode impute  -> one-hot
    high_cardinality -> mode impute  -> one-hot, rare titles grouped (min_frequency)
    """
    parts = []
    if roles["numeric"]:
        parts.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                roles["numeric"],
            )
        )
    if roles["ordinal"]:
        parts.append(
            (
                "ordinal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[EDUCATION_ORDER] * len(roles["ordinal"]),
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                roles["ordinal"],
            )
        )
    if roles["low_cardinality"]:
        parts.append(
            (
                "low_cardinality",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                roles["low_cardinality"],
            )
        )
    if roles["high_cardinality"]:
        parts.append(
            (
                "high_cardinality",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=JOB_TITLE_MIN_FREQUENCY,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                roles["high_cardinality"],
            )
        )
    if not parts:
        raise ValueError("No usable feature columns found.")
    return ColumnTransformer(parts)
