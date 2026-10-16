import pandas as pd

from api.repositories.model_repository import ModelRepository
from api.schemas.prediction import RentRequest
from ml.preprocessing.preprocess import FEATURES


class InvalidInputError(ValueError):
    pass


class PredictionService:
    """Business logic: validates against known categories and runs the model."""

    def __init__(self, repo: ModelRepository):
        self.repo = repo

    def get_options(self):
        meta = self.repo.get_metadata()
        return {k: meta[k] for k in ("cities", "furnishing", "localities")}

    def predict(self, req: RentRequest):
        meta = self.repo.get_metadata()
        if req.city not in meta["cities"]:
            raise InvalidInputError(f"Unknown city: {req.city}")
        if req.furnishing not in meta["furnishing"]:
            raise InvalidInputError(f"Unknown furnishing: {req.furnishing}")

        row = pd.DataFrame([req.model_dump()])[FEATURES]
        rent = float(self.repo.get_model().predict(row)[0])
        mae = meta["mae"]
        return {
            "rent": round(rent, -2),
            "low": round(max(rent - mae, 0), -2),
            "high": round(rent + mae, -2),
        }
