from fastapi import APIRouter

from api.controllers.prediction_controller import PredictionController
from api.repositories.model_repository import ModelRepository
from api.schemas.prediction import RentRequest, RentResponse
from api.services.prediction_service import PredictionService

controller = PredictionController(PredictionService(ModelRepository()))

router = APIRouter(prefix="/api")


@router.get("/options")
def options():
    return controller.options()


@router.post("/predict", response_model=RentResponse)
def predict(req: RentRequest):
    return controller.predict(req)
