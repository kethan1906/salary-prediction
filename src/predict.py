"""Predict a salary for one employee.

CLI example:
  python -m src.predict --age 32 --gender Male --education "Master's" \
      --job-title "Data Scientist" --years 6
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import pandas as pd
import sklearn

from src.config import EDUCATION_MAP, EDUCATION_ORDER, GENDER_VALUES
from src.data_loader import normalize_column_name
from src.model_loader import load_metadata, load_model


class ValidationError(ValueError):
    """The input cannot be used for prediction."""


def _number(data: dict, key: str, low: float, high: float) -> float:
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        raise ValidationError(f"{key} must be a number") from None
    if not math.isfinite(value) or not (low <= value <= high):
        raise ValidationError(f"{key} must be between {low:g} and {high:g}")
    return value


def validate_input(employee: dict, metadata: dict) -> tuple[dict, list[str]]:
    """Return (clean_row, warnings). Raises ValidationError for unusable input."""
    data = {normalize_column_name(k): v for k, v in employee.items()}
    missing = [c for c in metadata["raw_input_columns"] if data.get(c) in (None, "")]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")
    warns: list[str] = []

    age = _number(data, "Age", 16, 80)
    years = _number(data, "Years_of_Experience", 0, 60)
    if years > age:
        raise ValidationError("Years_of_Experience cannot exceed Age")
    if years > age - 16:
        warns.append("Experience is unusually high for this age (started working before 16).")

    gender = str(data["Gender"]).strip().capitalize()
    if gender not in GENDER_VALUES:
        raise ValidationError(f"Gender must be one of {GENDER_VALUES}")

    education = EDUCATION_MAP.get(str(data["Education_Level"]).strip().lower())
    if education is None:
        raise ValidationError(f"Education_Level must be one of {EDUCATION_ORDER}")

    title = re.sub(r"\s+", " ", str(data["Job_Title"])).strip()
    if not title or len(title) > 100:
        raise ValidationError("Job_Title must be 1-100 characters")
    known = {t.lower(): t for t in metadata["known_categories"]["Job_Title"]}
    if title.lower() in known:
        title = known[title.lower()]
    else:
        warns.append("This job title was not in the training data; the model treats it as a rare "
                     "title and relies mostly on seniority keywords, experience and age.")

    for col, value in (("Age", age), ("Years_of_Experience", years)):
        low, high = metadata["training_ranges"][col]
        if not (low <= value <= high):
            warns.append(f"{col}={value:g} is outside the training range [{low:g}, {high:g}]; "
                         "the estimate is an extrapolation and may be unreliable.")

    clean = {"Age": age, "Gender": gender, "Education_Level": education,
             "Job_Title": title, "Years_of_Experience": years}
    return clean, warns


def predict_detailed(employee: dict, model_path=None, metadata_path=None) -> dict:
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)
    warns: list[str] = []
    if metadata.get("sklearn_version") != sklearn.__version__:
        warns.append(f"Model was trained with scikit-learn {metadata.get('sklearn_version')} "
                     f"but {sklearn.__version__} is installed; retrain if results look wrong.")
    clean, input_warns = validate_input(employee, metadata)
    salary = float(model.predict(pd.DataFrame([clean]))[0])
    return {"prediction": salary, "warnings": warns + input_warns, "input": clean}


def predict_salary(employee_data: dict, model_path=None, metadata_path=None) -> float:
    result = predict_detailed(employee_data, model_path, metadata_path)
    for w in result["warnings"]:
        warnings.warn(w, UserWarning, stacklevel=2)
    return result["prediction"]


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a salary.")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--gender", required=True)
    p.add_argument("--education", required=True)
    p.add_argument("--job-title", required=True)
    p.add_argument("--years", type=float, required=True)
    a = p.parse_args()
    try:
        r = predict_detailed({"Age": a.age, "Gender": a.gender, "Education_Level": a.education,
                              "Job_Title": a.job_title, "Years_of_Experience": a.years})
    except (ValidationError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(f"Predicted salary: {r['prediction']:,.0f}")
    for w in r["warnings"]:
        print(f"Warning: {w}")


if __name__ == "__main__":
    main()
