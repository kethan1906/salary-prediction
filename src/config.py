"""Central configuration: paths, split settings and column conventions."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "Salary_Data.csv"
MODEL_PATH = ROOT_DIR / "models" / "salary_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COLUMN = "Salary"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Raw input columns the model expects (names after normalisation).
RAW_FEATURES = ["Age", "Gender", "Education_Level", "Job_Title", "Years_of_Experience"]

# Cleaning rules
EDUCATION_ORDER = ["High School", "Bachelor's", "Master's", "PhD"]
EDUCATION_MAP = {
    "high school": "High School",
    "bachelor's": "Bachelor's",
    "bachelors": "Bachelor's",
    "bachelor's degree": "Bachelor's",
    "master's": "Master's",
    "masters": "Master's",
    "master's degree": "Master's",
    "phd": "PhD",
}
GENDER_VALUES = ["Male", "Female", "Other"]
# The lowest plausible annual salary kept as training data. In this dataset the
# four values below it are 350-579 while the next lowest is 25,000, i.e. they are
# almost certainly entry errors (or monthly figures), not annual salaries.
MIN_PLAUSIBLE_SALARY = 10_000

# Feature roles used by the preprocessor (only columns that exist are used)
NUMERIC_FEATURES = ["Age", "Years_of_Experience", "Experience_Squared"]
ORDINAL_FEATURES = ["Education_Level"]
LOW_CARDINALITY_FEATURES = ["Gender", "Title_Level", "Career_Stage"]
HIGH_CARDINALITY_FEATURES = ["Job_Title"]
JOB_TITLE_MIN_FREQUENCY = 5  # rarer titles share one "infrequent" bucket

BOOTSTRAP_SAMPLES = 1000
PERMUTATION_REPEATS = 30
