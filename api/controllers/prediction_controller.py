from fastapi import HTTPException

from api.schemas.prediction import RentRequest
from api.services.prediction_service import InvalidInputError, PredictionService


class PredictionController:
    """Handles request/response concerns and maps service errors to HTTP errors."""

    def __init__(self, service: PredictionService):
        self.service = service

    def options(self):
        return self.service.get_options()

    def predict(self, req: RentRequest):
        try:
            return self.service.predict(req)
        except InvalidInputError as e:
            raise HTTPException(status_code=422, detail=str(e))
