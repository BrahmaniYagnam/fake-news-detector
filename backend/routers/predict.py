from fastapi import APIRouter, HTTPException

from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.model_service import ModelService
from backend.utils.text import clean_text

router = APIRouter()
model_service = ModelService()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    text = clean_text(request.text)
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    try:
        prediction_result = model_service.predict(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        prediction=prediction_result["prediction"],
        confidence=prediction_result["confidence"],
        probabilities=prediction_result["probabilities"],
        important_keywords=prediction_result.get("important_keywords", []),
    )
