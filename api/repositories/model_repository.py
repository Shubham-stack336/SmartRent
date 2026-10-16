import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "ml/models/rent_model.pkl"
META_PATH = ROOT / "ml/models/metadata.json"


class ModelRepository:
    """Data-access layer: loads the trained model and its metadata from disk."""

    def __init__(self):
        self._model = joblib.load(MODEL_PATH)
        self._meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def get_model(self):
        return self._model

    def get_metadata(self):
        return self._meta
