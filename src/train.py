"""Train, tune, compare and persist the salary model.

Run:  python -m src.train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import (
    BOOTSTRAP_SAMPLES, CV_FOLDS, DATA_PATH, FIGURES_DIR, METADATA_PATH, MODEL_PATH,
    PERMUTATION_REPEATS, RANDOM_STATE, RAW_FEATURES, REPORTS_DIR, TARGET_COLUMN, TEST_SIZE,
)
from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.models import get_candidate_models, get_param_grids
from src.preprocessing import build_preprocessor, select_features
from src.reporting import model_figures

SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def make_pipeline(model, roles) -> Pipeline:
    """raw columns -> feature engineering -> preprocessing -> model (one saved object)."""
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("preprocessor", build_preprocessor(roles)),
            ("model", model),
        ]
    )


def _metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _bootstrap_ci(y_true, y_pred, n: int) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse, r2 = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        rmse.append(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
        r2.append(r2_score(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(rmse, [2.5, 97.5]), np.percentile(r2, [2.5, 97.5])
    return {"RMSE_95CI": [float(lo[0]), float(lo[1])], "R2_95CI": [float(hi[0]), float(hi[1])],
            "bootstrap_samples": n}


def _cv_summary(scores: dict) -> dict:
    return {
        "RMSE": float(-np.mean(scores["test_rmse"])),
        "RMSE_std": float(np.std(scores["test_rmse"])),
        "MAE": float(-np.mean(scores["test_mae"])),
        "R2": float(np.mean(scores["test_r2"])),
    }


def run_training(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    metadata_path=METADATA_PATH,
    reports_dir=REPORTS_DIR,
    figures_dir=FIGURES_DIR,
    param_grids: dict | None = None,
    make_figures: bool = True,
    n_jobs: int = -1,
) -> dict:
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    reports_dir, figures_dir = Path(reports_dir), Path(figures_dir)
    grids = param_grids if param_grids is not None else get_param_grids()

    df, cleaning = load_data(data_path, with_report=True)
    X, y = df[RAW_FEATURES], df[TARGET_COLUMN]
    roles = select_features(engineer_features(X))

    # Duplicates were removed during cleaning, so the same row cannot be in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    shared = int(
        pd.concat([X_train, y_train], axis=1).merge(
            pd.concat([X_test, y_test], axis=1), how="inner"
        ).shape[0]
    )

    # Shuffled folds, same folds for every model.
    cv = KFold(n_splits=min(CV_FOLDS, len(X_train)), shuffle=True, random_state=RANDOM_STATE)

    results, searches = {}, {}
    for name, estimator in get_candidate_models().items():
        default_scores = cross_validate(
            make_pipeline(estimator, roles), X_train, y_train, cv=cv, scoring=SCORING, n_jobs=n_jobs
        )
        grid = grids.get(name, {})
        search = GridSearchCV(
            make_pipeline(get_candidate_models()[name], roles),
            param_grid=grid, cv=cv, scoring=SCORING, refit="rmse", n_jobs=n_jobs,
        ).fit(X_train, y_train)
        i = search.best_index_
        results[name] = {
            "default_settings_cv": _cv_summary(default_scores),
            "tuned_cv": {
                "RMSE": float(-search.cv_results_["mean_test_rmse"][i]),
                "RMSE_std": float(search.cv_results_["std_test_rmse"][i]),
                "MAE": float(-search.cv_results_["mean_test_mae"][i]),
                "R2": float(search.cv_results_["mean_test_r2"][i]),
            },
            "grid_combinations": int(len(search.cv_results_["params"])),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        }
        searches[name] = search

    best_name = min(results, key=lambda n: results[n]["tuned_cv"]["RMSE"])
    final_pipeline = searches[best_name].best_estimator_  # already refit on the full training split

    # The test split is evaluated exactly once, here.
    test_pred = final_pipeline.predict(X_test)
    test_metrics = _metrics(y_test, test_pred)
    test_ci = _bootstrap_ci(y_test, test_pred, BOOTSTRAP_SAMPLES)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_metrics = _metrics(y_test, baseline.predict(X_test))

    # Robustness check: rows that are not exact duplicates can still share identical features
    # (different salary). Measure how common that is across the split, and re-score the selected
    # model with grouped folds where identical-feature rows always stay together (stricter).
    keys = lambda frame: frame.apply(lambda r: "|".join(map(str, r)), axis=1)  # noqa: E731
    train_keys, test_keys = keys(X_train), keys(X_test)
    grouped_scores = cross_validate(
        clone(final_pipeline), X_train, y_train, cv=GroupKFold(n_splits=cv.get_n_splits()),
        groups=pd.factorize(train_keys)[0], scoring=SCORING, n_jobs=n_jobs,
    )
    robustness = {
        "test_rows_with_identical_features_in_train": float(test_keys.isin(set(train_keys)).mean()),
        "grouped_cv_selected_model_on_train": _cv_summary(grouped_scores),
    }

    perm = permutation_importance(
        final_pipeline, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=n_jobs,
    )
    importance = (
        pd.DataFrame({"feature": RAW_FEATURES, "importance_mean": perm.importances_mean,
                      "importance_std": perm.importances_std})
        .sort_values("importance_mean", ascending=False).reset_index(drop=True)
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    title_counts = X_train["Job_Title"].value_counts()
    metadata = {
        "sklearn_version": sklearn.__version__,
        "raw_input_columns": RAW_FEATURES,
        "selected_model": best_name,
        "best_params": results[best_name]["best_params"],
        "training_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())]
                            for c in ("Age", "Years_of_Experience")},
        "known_categories": {
            "Gender": sorted(X_train["Gender"].dropna().unique()),
            "Education_Level": [e for e in ["High School", "Bachelor's", "Master's", "PhD"]
                                if e in set(X_train["Education_Level"].dropna())],
            "Job_Title": sorted(title_counts.index),
        },
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_R2": test_metrics["R2"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    comparison = pd.DataFrame(
        {
            "model": list(results),
            "cv_rmse_default": [r["default_settings_cv"]["RMSE"] for r in results.values()],
            "cv_rmse_tuned": [r["tuned_cv"]["RMSE"] for r in results.values()],
            "cv_mae_tuned": [r["tuned_cv"]["MAE"] for r in results.values()],
            "cv_r2_tuned": [r["tuned_cv"]["R2"] for r in results.values()],
        }
    )
    comparison["selected"] = comparison["model"] == best_name

    metrics = {
        "dataset": {
            **cleaning,
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "exact_duplicate_rows_shared_between_train_and_test": shared,
        },
        "settings": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                     "cv_folds": cv.get_n_splits(), "cv_shuffle": True, "raw_features": RAW_FEATURES,
                     "feature_roles_after_engineering": roles},
        "models": results,
        "selected_model": best_name,
        "test_metrics_selected_model": test_metrics,
        "test_bootstrap_ci_selected_model": test_ci,
        "reference_mean_baseline_test_metrics": baseline_metrics,
        "robustness": robustness,
        "permutation_importance_test": importance.round(1).to_dict(orient="records"),
        "sklearn_version": sklearn.__version__,
        "notes": ("Tuned CV scores are selected on the same folds and are slightly optimistic; "
                  "the test metrics come from rows never used for tuning or selection."),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    if make_figures:
        model_figures(comparison, y_test, test_pred, importance, figures_dir)
    return metrics


def _print_summary(m: dict) -> None:
    d = m["dataset"]
    print(f"Rows: {d['rows_raw']} raw -> {d['rows_clean']} clean unique | train {d['rows_train']} / test {d['rows_test']}"
          f" | duplicate rows shared by train and test: {d['exact_duplicate_rows_shared_between_train_and_test']}")
    print(f"\n{m['settings']['cv_folds']}-fold CV on the training split:")
    for n, r in m["models"].items():
        t, o = r["tuned_cv"], r["default_settings_cv"]
        print(f"  {n:<18} default RMSE={o['RMSE']:>9,.0f} | tuned RMSE={t['RMSE']:>9,.0f} MAE={t['MAE']:>9,.0f} R2={t['R2']:.3f}"
              f"  params={r['best_params']}")
    t, b, ci = m["test_metrics_selected_model"], m["reference_mean_baseline_test_metrics"], m["test_bootstrap_ci_selected_model"]
    print(f"\nSelected: {m['selected_model']}")
    print(f"Held-out test ({d['rows_test']} rows): RMSE={t['RMSE']:,.0f} (95% CI {ci['RMSE_95CI'][0]:,.0f}-{ci['RMSE_95CI'][1]:,.0f}) "
          f"MAE={t['MAE']:,.0f} R2={t['R2']:.3f} (95% CI {ci['R2_95CI'][0]:.3f}-{ci['R2_95CI'][1]:.3f})")
    print(f"Mean-predictor baseline: RMSE={b['RMSE']:,.0f} R2={b['R2']:.3f}")
    rb = m["robustness"]
    print(f"Robustness: {rb['test_rows_with_identical_features_in_train']:.1%} of test rows have a train row with identical features; "
          f"grouped CV (selected model) RMSE={rb['grouped_cv_selected_model_on_train']['RMSE']:,.0f} R2={rb['grouped_cv_selected_model_on_train']['R2']:.3f}")
    print("Permutation importance (RMSE increase):",
          ", ".join(f"{r['feature']}={r['importance_mean']:,.0f}" for r in m["permutation_importance_test"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the salary prediction pipeline.")
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()
    _print_summary(run_training(data_path=args.data))
    print(f"\nSaved model -> {MODEL_PATH}\nSaved reports -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()
