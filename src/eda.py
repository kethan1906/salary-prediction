"""Exploratory data analysis: data-quality report, summary statistics, figures.

Run:  python -m src.eda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DATA_PATH, EDUCATION_ORDER, FIGURES_DIR, REPORTS_DIR
from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.reporting import eda_figures


def _group_stats(df, col, order=None):
    g = df.groupby(col)["Salary"].agg(["count", "median", "mean"]).round(0)
    if order:
        g = g.reindex([o for o in order if o in g.index])
    return {str(k): {"rows": int(v["count"]), "median": float(v["median"]), "mean": float(v["mean"])}
            for k, v in g.iterrows()}


def run_eda(data_path=DATA_PATH, reports_dir=REPORTS_DIR, figures_dir=FIGURES_DIR) -> dict:
    reports_dir, figures_dir = Path(reports_dir), Path(figures_dir)
    df, cleaning = load_data(data_path, with_report=True)
    eng = engineer_features(df)

    summary = {
        "data_cleaning": cleaning,
        "salary": {k: float(v) for k, v in df["Salary"].describe().round(1).items()},
        "correlation": df[["Age", "Years_of_Experience", "Salary"]].corr().round(3).to_dict(),
        "distinct_job_titles": int(df["Job_Title"].nunique()),
        "job_titles_with_fewer_than_5_rows": int((df["Job_Title"].value_counts() < 5).sum()),
        "by_education": _group_stats(df, "Education_Level", EDUCATION_ORDER),
        "by_gender": _group_stats(df, "Gender"),
        "by_title_level": _group_stats(eng, "Title_Level"),
        "note": ("Group statistics are descriptive. They are not controlled for experience, "
                 "education or job title and must not be read as causal effects."),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_cleaning.json").write_text(json.dumps(cleaning, indent=2))
    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    summary["figures"] = eda_figures(df, figures_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()
    s = run_eda(args.data)
    c = s["data_cleaning"]
    print(f"Raw rows {c['rows_raw']} -> clean unique rows {c['rows_clean']} "
          f"(dropped: {c['dropped_exact_duplicates']} duplicates, {c['dropped_missing_salary']} missing salary, "
          f"{c['dropped_implausible_salary']} implausible salary)")
    print("Correlation with Salary:", {k: v for k, v in s["correlation"]["Salary"].items()})
    print("Figures:", ", ".join(s["figures"]))


if __name__ == "__main__":
    main()
