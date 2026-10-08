"""The three candidate regressors, their original fixed settings and tuning grids."""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import RANDOM_STATE


def get_candidate_models() -> dict:
    """Original fixed settings (used as the 'untuned' reference)."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE,
        ),
    }


def get_param_grids() -> dict:
    """GridSearchCV grids (keys are pipeline parameter names)."""
    return {
        "linear_regression": {},  # no hyperparameters worth tuning
        "random_forest": {
            "model__n_estimators": [300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "gradient_boosting": {
            "model__n_estimators": [300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 4, 5],
            "model__min_samples_leaf": [1, 3],
        },
    }
