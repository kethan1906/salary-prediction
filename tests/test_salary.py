import json

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PATH, RAW_FEATURES
from src.data_loader import clean_data, load_data
from src.feature_engineering import engineer_features, title_level
from src.preprocessing import build_preprocessor, select_features
from src.train import run_training

FAST_GRIDS = {
    "linear_regression": {},
    "random_forest": {"model__n_estimators": [50], "model__max_depth": [10]},
    "gradient_boosting": {"model__n_estimators": [50], "model__max_depth": [3]},
}


def raw_frame(rows):
    return pd.DataFrame(
        rows, columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]
    )


# ---------- data cleaning ----------
def test_cleaning_rules_and_report():
    raw = raw_frame([
        [30, "Male", "Bachelor's Degree", "Data  Analyst", 5, 60000],
        [30, "Male", "Bachelor's", "Data Analyst", 5, 60000],      # duplicate after normalisation
        [28, "Female", "phD", "Engineer", 3, 70000],
        [25, "female", "Master's Degree", "Engineer", 1, 350],      # implausible salary
        [40, "Male", "PhD", "Manager", 15, np.nan],                 # missing salary
        [33, "Male", None, "Developer", 7, 100000],                 # missing feature -> kept
    ])
    df, rep = clean_data(raw)
    assert rep["rows_raw"] == 6 and rep["rows_clean"] == 3
    assert rep["dropped_missing_salary"] == 1
    assert rep["dropped_implausible_salary"] == 1
    assert rep["dropped_exact_duplicates"] == 1
    assert rep["rows_with_missing_features_kept_for_imputation"] == 1
    assert set(df["Education_Level"].dropna()) == {"Bachelor's", "PhD"}
    assert "Data Analyst" in set(df["Job_Title"])  # whitespace collapsed


def test_cleaning_missing_columns_and_file(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        clean_data(pd.DataFrame({"a": [1]}))
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_real_dataset_cleaning_counts():
    df, rep = load_data(DATA_PATH, with_report=True)
    assert rep["rows_raw"] == 6704
    assert rep["dropped_exact_duplicates"] > 4000  # heavy duplication in the source file
    assert not df.duplicated().any()
    assert df["Salary"].min() >= 10_000
    assert set(df["Education_Level"].dropna()) <= {"High School", "Bachelor's", "Master's", "PhD"}


# ---------- feature engineering / preprocessing ----------
@pytest.mark.parametrize("title,level", [
    ("Chief Technology Officer", "Executive"), ("CEO", "Executive"),
    ("Director of Sales", "Director"), ("Software Engineer Manager", "Manager"),
    ("Senior Software Engineer", "Senior"), ("Junior HR Coordinator", "Junior"),
    ("Data Scientist", "Standard"),
])
def test_title_level(title, level):
    assert title_level(title) == level


def test_engineer_features_no_mutation():
    df = pd.DataFrame({"Years_of_Experience": [0, 5, 20], "Job_Title": ["A", "Senior B", "C Manager"]})
    out = engineer_features(df)
    assert "Experience_Squared" not in df.columns
    assert out["Experience_Squared"].tolist() == [0, 25, 400]
    assert out["Career_Stage"].tolist() == ["Early_Career", "Mid_Career", "Senior"]
    assert out["Title_Level"].tolist() == ["Standard", "Senior", "Manager"]


def test_preprocessor_handles_missing_ordinal_and_unseen():
    df, _ = load_data(DATA_PATH, with_report=True)
    X = df[RAW_FEATURES]
    eng = engineer_features(X)
    roles = select_features(eng)
    assert roles["ordinal"] == ["Education_Level"] and roles["high_cardinality"] == ["Job_Title"]
    pre = build_preprocessor(roles).fit(eng)
    out = pre.transform(eng)
    assert not np.isnan(out).any()  # the row with missing education was imputed
    unseen = eng.iloc[[0]].copy()
    unseen["Job_Title"] = "Totally New Title"
    assert pre.transform(unseen).shape == (1, out.shape[1])


def test_ordinal_education_is_ordered():
    df = pd.DataFrame({"Education_Level": ["High School", "Bachelor's", "Master's", "PhD"]})
    roles = {"numeric": [], "ordinal": ["Education_Level"], "low_cardinality": [], "high_cardinality": []}
    values = build_preprocessor(roles).fit_transform(df).ravel()
    assert list(values) == sorted(values)


# ---------- training end to end (small grids so the test is quick) ----------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    paths = dict(model_path=out / "m.joblib", metadata_path=out / "meta.json")
    metrics = run_training(
        DATA_PATH, **paths, reports_dir=out / "reports", figures_dir=out / "figs",
        param_grids=FAST_GRIDS, n_jobs=1,
    )
    return metrics, paths, out


def test_training_outputs(trained):
    metrics, paths, out = trained
    assert paths["model_path"].exists() and paths["metadata_path"].exists()
    for f in ("metrics.json", "model_comparison.csv", "feature_importance.csv"):
        assert (out / "reports" / f).exists()
    assert len(list((out / "figs").glob("*.png"))) == 4
    d = metrics["dataset"]
    assert d["exact_duplicate_rows_shared_between_train_and_test"] == 0
    assert d["rows_train"] + d["rows_test"] == d["rows_clean"]


def test_selection_and_sanity(trained):
    metrics, _, _ = trained
    models = metrics["models"]
    assert set(models) == {"linear_regression", "random_forest", "gradient_boosting"}
    assert metrics["selected_model"] == min(models, key=lambda n: models[n]["tuned_cv"]["RMSE"])
    t, b = metrics["test_metrics_selected_model"], metrics["reference_mean_baseline_test_metrics"]
    assert t["RMSE"] < b["RMSE"] and t["R2"] > 0.5
    lo, hi = metrics["test_bootstrap_ci_selected_model"]["RMSE_95CI"]
    assert lo <= t["RMSE"] <= hi
    rb = metrics["robustness"]
    assert 0 <= rb["test_rows_with_identical_features_in_train"] <= 1
    assert rb["grouped_cv_selected_model_on_train"]["RMSE"] > 0
    imp = {r["feature"]: r["importance_mean"] for r in metrics["permutation_importance_test"]}
    assert set(imp) == set(RAW_FEATURES)
    assert imp["Years_of_Experience"] > imp["Gender"]


def test_saved_pipeline_accepts_raw_columns(trained):
    from src.model_loader import load_model
    _, paths, _ = trained
    model = load_model(paths["model_path"])
    row = pd.DataFrame([{"Age": 30, "Gender": "Male", "Education_Level": "Master's",
                         "Job_Title": "Data Scientist", "Years_of_Experience": 5}])
    assert np.isfinite(model.predict(row)[0])


def test_training_is_reproducible(trained, tmp_path):
    metrics, _, _ = trained
    again = run_training(
        DATA_PATH, tmp_path / "m.joblib", tmp_path / "meta.json", tmp_path / "r", tmp_path / "f",
        param_grids=FAST_GRIDS, make_figures=False, n_jobs=1,
    )
    assert again["test_metrics_selected_model"] == metrics["test_metrics_selected_model"]
    assert again["models"] == metrics["models"]


# ---------- prediction ----------
EMP = {"Age": 32, "Gender": "Male", "Education_Level": "Master's",
       "Job_Title": "Data Scientist", "Years_of_Experience": 6}


def test_predict_valid_and_case_insensitive(trained):
    from src.predict import predict_detailed
    _, p, _ = trained
    a = predict_detailed(EMP, p["model_path"], p["metadata_path"])
    b = predict_detailed({**EMP, "Job_Title": "  data   scientist "}, p["model_path"], p["metadata_path"])
    assert a["prediction"] == b["prediction"] and 20_000 < a["prediction"] < 300_000
    assert a["warnings"] == []


@pytest.mark.parametrize("change", [
    {"Age": 10}, {"Age": "abc"}, {"Years_of_Experience": -1}, {"Years_of_Experience": 50},
    {"Gender": "robot"}, {"Education_Level": "Kindergarten"}, {"Job_Title": ""}, {"Age": float("nan")},
])
def test_predict_rejects_bad_input(trained, change):
    from src.predict import ValidationError, predict_detailed
    _, p, _ = trained
    with pytest.raises(ValidationError):
        predict_detailed({**EMP, **change}, p["model_path"], p["metadata_path"])


def test_predict_missing_field_and_warnings(trained):
    from src.predict import ValidationError, predict_detailed
    _, p, _ = trained
    with pytest.raises(ValidationError, match="Missing"):
        predict_detailed({"Age": 30}, p["model_path"], p["metadata_path"])
    r = predict_detailed({**EMP, "Job_Title": "Chief Llama Officer"}, p["model_path"], p["metadata_path"])
    assert any("not in the training data" in w for w in r["warnings"])
    r = predict_detailed({**EMP, "Age": 79, "Years_of_Experience": 55}, p["model_path"], p["metadata_path"])
    assert any("outside the training range" in w for w in r["warnings"])


# ---------- web app ----------
@pytest.fixture()
def client(trained):
    from app import create_app
    _, p, _ = trained
    app = create_app(p["model_path"], p["metadata_path"])
    app.testing = True
    return app.test_client()


def test_app_pages_and_options(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json() == {"status": "ok"}
    o = client.get("/api/options").get_json()
    assert "Male" in o["genders"] and o["education_levels"][-1] == "PhD" and len(o["job_titles"]) > 50


def test_app_predict(client):
    r = client.post("/api/predict", json=EMP)
    body = r.get_json()
    assert r.status_code == 200 and body["predicted_salary"] > 0 and body["typical_error_mae"] > 0
    assert client.post("/api/predict", json={**EMP, "Age": 5}).status_code == 400
    assert client.post("/api/predict", data="not json").status_code == 400
    assert client.post("/api/predict", json=[1, 2]).status_code == 400
    assert client.get("/nope").get_json() == {"error": "Not found"}


def test_app_without_model(tmp_path):
    from app import create_app
    c = create_app(tmp_path / "x.joblib", tmp_path / "x.json").test_client()
    assert c.get("/api/options").status_code == 503
    assert c.post("/api/predict", json=EMP).status_code == 503
