"""Load the persisted pipeline and its metadata (cached)."""
import json
from functools import lru_cache
from pathlib import Path

import joblib

from src.config import METADATA_PATH, MODEL_PATH


@lru_cache(maxsize=4)
def _load_model_cached(path: str):
    return joblib.load(path)


def load_model(path: Path | str | None = None):
    path = Path(path) if path is not None else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m src.train")
    return _load_model_cached(str(path))


def load_metadata(path: Path | str | None = None) -> dict:
    path = Path(path) if path is not None else METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"No model metadata at {path}. Run: python -m src.train")
    return json.loads(path.read_text())
