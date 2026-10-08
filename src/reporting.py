"""Matplotlib figures for EDA and model evaluation (saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import EDUCATION_ORDER  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#1f4e79", "#e08a00"


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def eda_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    saved = []

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(df["Salary"], bins=40, color=BLUE)
    ax.set(title="Salary distribution (cleaned, de-duplicated data)", xlabel="Salary", ylabel="Rows")
    saved.append(_save(fig, out_dir / "salary_distribution.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(df["Years_of_Experience"], df["Salary"], s=8, alpha=0.4, color=BLUE)
    ok = df[["Years_of_Experience", "Salary"]].dropna()
    slope, intercept = np.polyfit(ok["Years_of_Experience"], ok["Salary"], 1)
    xs = np.linspace(ok["Years_of_Experience"].min(), ok["Years_of_Experience"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color=ORANGE, label="linear trend")
    ax.set(title="Salary vs years of experience", xlabel="Years of experience", ylabel="Salary")
    ax.legend()
    saved.append(_save(fig, out_dir / "salary_vs_experience.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    groups = [(e, df.loc[df["Education_Level"] == e, "Salary"]) for e in EDUCATION_ORDER]
    ax.boxplot([g for _, g in groups], tick_labels=[f"{e}\n(n={len(g)})" for e, g in groups])
    ax.set(title="Salary by education level", ylabel="Salary")
    saved.append(_save(fig, out_dir / "salary_by_education.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    genders = [g for g in ["Male", "Female", "Other"] if (df["Gender"] == g).any()]
    ax.boxplot([df.loc[df["Gender"] == g, "Salary"] for g in genders],
               tick_labels=[f"{g}\n(n={(df['Gender'] == g).sum()})" for g in genders])
    ax.set(title="Salary by gender (descriptive only; not controlled for other factors)", ylabel="Salary")
    ax.title.set_fontsize(9)
    saved.append(_save(fig, out_dir / "salary_by_gender.png"))

    top = df["Job_Title"].value_counts().head(12).index
    med = df[df["Job_Title"].isin(top)].groupby("Job_Title")["Salary"].median().sort_values()
    counts = df["Job_Title"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([f"{t} (n={counts[t]})" for t in med.index], med.values, color=BLUE)
    ax.set(title="Median salary, 12 most common job titles", xlabel="Median salary")
    saved.append(_save(fig, out_dir / "median_salary_top_titles.png"))

    num = df[["Age", "Years_of_Experience", "Salary"]].corr()
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(num.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3), num.columns, rotation=30, ha="right")
    ax.set_yticks(range(3), num.columns)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{num.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation (Pearson)")
    saved.append(_save(fig, out_dir / "correlation.png"))
    return saved


def model_figures(comparison: pd.DataFrame, y_true, y_pred, importance: pd.DataFrame, out_dir: Path) -> list[str]:
    saved = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(comparison))
    ax.bar(x - 0.2, comparison["cv_rmse_default"], 0.4, label="original fixed settings", color="#9bb7d4")
    ax.bar(x + 0.2, comparison["cv_rmse_tuned"], 0.4, label="GridSearchCV-tuned", color=BLUE)
    ax.set_xticks(x, comparison["model"], rotation=15)
    ax.set(ylabel="5-fold CV RMSE (lower is better)", title="Model comparison on the training split")
    ax.legend()
    saved.append(_save(fig, out_dir / "model_comparison.png"))

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color=BLUE)
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, color=ORANGE, label="perfect prediction")
    ax.set(title="Held-out test set: predicted vs actual", xlabel="Actual salary", ylabel="Predicted salary")
    ax.legend()
    saved.append(_save(fig, out_dir / "predicted_vs_actual.png"))

    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=35, color=BLUE)
    ax.axvline(0, color=ORANGE)
    ax.set(title="Held-out test set: residuals (actual - predicted)", xlabel="Residual", ylabel="Rows")
    saved.append(_save(fig, out_dir / "residuals.png"))

    imp = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color=BLUE)
    ax.set(title="Permutation importance (test set)", xlabel="Increase in RMSE when the column is shuffled")
    saved.append(_save(fig, out_dir / "feature_importance.png"))
    return saved
