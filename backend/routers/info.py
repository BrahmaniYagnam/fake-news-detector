from fastapi import APIRouter

from backend.schemas.prediction import ModelInfoResponse
from backend.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info() -> ModelInfoResponse:
    info = model_service.get_info()
    return ModelInfoResponse(**info)
