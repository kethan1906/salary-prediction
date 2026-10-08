# Salary Prediction System

A supervised regression project (Python, pandas, scikit-learn, Flask) that predicts salary from age, gender, education level, job title and years of experience. It covers the full workflow: data cleaning, EDA, feature engineering, three-model comparison with cross-validation and GridSearchCV, held-out evaluation, feature-importance analysis, model persistence and a small web demo.

## Dataset and attribution

`data/raw/Salary_Data.csv` - 6,704 rows, 6 columns (`Age, Gender, Education Level, Job Title, Years of Experience, Salary`).

- **Source:** the public Kaggle dataset *Salary_Data* by **mohithsairamreddy** - https://www.kaggle.com/datasets/mohithsairamreddy/salary-data. Its page describes the data as 6,704 points collected from surveys, job-posting sites and other public sources.
- The bundled copy was downloaded unchanged from a public GitHub mirror (https://github.com/Ravi506051/Salary_Prediction-main).
- **Licence / provenance:** not verified here. Check the Kaggle page before publishing this repository; if redistribution is not allowed, delete the CSV and download it yourself to the same path. The Kaggle description is also inconsistent about whether `Salary` is monthly or annual (values look annual), so this project treats salaries as relative "dataset units".

### Data quality (important)

| Issue | Handling |
| --- | --- |
| 4,912 of 6,704 rows are **exact duplicates** (only ~1,790 unique rows) | Dropped **before** the train/test split so identical rows cannot appear on both sides |
| 5 rows without a salary | Dropped (nothing to learn from) |
| 4 salaries of 350-579 (next lowest is 25,000) | Dropped as implausible (< 10,000) |
| Inconsistent education labels ("Bachelor's" vs "Bachelor's Degree", "phD") | Normalised to High School / Bachelor's / Master's / PhD |
| 1 row with missing education | Kept; imputed with the mode inside the pipeline |
| 192 distinct job titles, most with few rows | Rare titles grouped (< 5 rows) + a title-seniority feature |

Result: **1,783 clean unique rows** (1,426 train / 357 test). Counts are in `reports/data_cleaning.json`.

Even after removing exact duplicates, 19% of test rows have a training row with identical features (and a different salary), which makes the held-out score somewhat optimistic; see Results.

## Pipeline

```
raw CSV -> clean (labels, missing, implausible, de-duplicate) -> 80/20 split (random_state=42)
 saved sklearn Pipeline:
   feature engineering (Experience_Squared, Career_Stage, Title_Level from job-title keywords)
   -> numeric: median impute + scale | education: ordered encoding (High School < ... < PhD)
      gender/seniority/career stage: one-hot | job title: one-hot, rare titles grouped
   -> model
 for each of Linear Regression / Random Forest / Gradient Boosting:
   5-fold shuffled CV with the original fixed settings, and GridSearchCV (same folds)
 lowest tuned CV RMSE wins -> evaluated ONCE on the held-out test split
 -> permutation importance, bootstrap confidence intervals, figures, saved model + metadata
```

Because feature engineering is part of the saved pipeline, prediction takes raw columns and cannot drift from training.

## Results (real data; `python -m src.train`)

5-fold CV on the 1,426 training rows (mean):

| Model | RMSE, original settings | RMSE, GridSearchCV-tuned | MAE (tuned) | R2 (tuned) |
| --- | --- | --- | --- | --- |
| Linear Regression | 18,105 | 18,105 (nothing to tune) | 13,141 | 0.874 |
| Random Forest | 15,978 | 14,962 | 9,959 | 0.914 |
| **Gradient Boosting (selected)** | 16,906 | **14,441** | **9,551** | **0.920** |

Tuning helped (about 6% lower RMSE for Random Forest, 15% for Gradient Boosting). Tuned CV scores are slightly optimistic because the same folds choose the settings. The Gradient Boosting grid was widened once (before the final run) after the best CV setting sat on the edge of the first grid; search spaces are in `src/models.py`. Selected: `learning_rate=0.2, max_depth=4, min_samples_leaf=1, n_estimators=300`.

**Held-out test set (357 rows, used once):**

| Metric | Value | 95% bootstrap CI |
| --- | --- | --- |
| RMSE | 14,942 | 12,602 - 17,219 |
| MAE | 9,398 | |
| R2 | 0.918 | 0.887 - 0.942 |

Always predicting the training mean gives RMSE 52,068 and R2 0.000 on the same rows.

**Stricter check:** because 19% of test rows share identical features with a training row, the selected model was also scored with *grouped* 5-fold CV on the training split (identical-feature rows kept together): RMSE 15,827, R2 0.904. Treat roughly **R2 0.90** as the more conservative estimate. All figures are in `reports/metrics.json`; exact values can shift slightly with other library versions. The numbers in this README and in `reports/` were produced with scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4. A clean install of `requirements.txt` with scikit-learn 1.9.1 / pandas 3.0.6 gave identical Linear Regression and Random Forest scores and the same selected model, but slightly different Gradient Boosting results (tuned CV RMSE 14,402; test RMSE 14,689, MAE 9,294, R2 0.920; grouped-CV R2 0.911). Differences of this size are version noise, not a change in conclusions.

### Feature importance (permutation, test set; RMSE increase when a column is shuffled)

Years_of_Experience 48.8k, Job_Title 20.2k, Age 7.2k, Education_Level 3.2k, Gender 0.1k (std 0.1k). Age and experience are strongly correlated (r = 0.94), so importance is shared between them and Age is understated. The model barely relies on gender; the raw median gap between genders in the data is descriptive and is not controlled for experience, education or title, so it is not evidence of a causal effect.

![importance](reports/figures/feature_importance.png)

### EDA highlights (`python -m src.eda`, figures in `reports/figures/`)

Salary correlates with years of experience (r = 0.82) and age (0.77). Median salary rises with education (High School 35k, Bachelor's 80k, Master's 127k, PhD 170k) and with title seniority (Junior 40k, Standard 95k, Manager 125k, Senior 145k, Director 160k). These are descriptive statistics, not causal effects.

![experience](reports/figures/salary_vs_experience.png) ![education](reports/figures/salary_by_education.png)
![compare](reports/figures/model_comparison.png) ![pred](reports/figures/predicted_vs_actual.png)

## Limitations

- The data are a public, partly duplicated file of unclear provenance; results show the pipeline works on this data, not that it predicts real-world pay.
- Salary units are unclear; no currency or period is claimed.
- Titles not in the training data are handled approximately (seniority keywords, experience, age); the web demo warns in that case. Inputs outside age 21-62 or 0-34 years of experience are extrapolation (tree models flatten out).
- Near-identical rows mean even the held-out score is somewhat optimistic (see grouped CV above).
- Rare gender category "Other" has only 7 rows.

## Setup and usage

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda        # data-quality report + EDA figures      (seconds)
python -m src.train      # tune, compare, evaluate, save model     (a few minutes on one CPU core)
python app.py            # web demo at http://127.0.0.1:5000
pytest                   # tests (about 30 s)

python -m src.predict --age 32 --gender Male --education "Master's" --job-title "Data Scientist" --years 6
```

`models/` is generated (git-ignored); run `python -m src.train` first. The saved `.joblib` is tied to the scikit-learn version recorded in `models/model_metadata.json`; retrain after upgrading, and only load model files you trained yourself.

Web API: `GET /api/options`, `POST /api/predict` (JSON with `Age, Gender, Education_Level, Job_Title, Years_of_Experience`), `GET /api/health`. Invalid input returns HTTP 400 with an error message.

## Layout

```
src/   config, data_loader, feature_engineering, preprocessing, models, train, eda,
       reporting, predict, model_loader
app.py templates/ static/        web demo
tests/                           30+ tests (cleaning, features, training, prediction, API)
reports/                         metrics.json, model_comparison.csv, feature_importance.csv,
                                 data_cleaning.json, eda_summary.json, figures/
data/raw/Salary_Data.csv  data/README.md
```
