from fastapi import APIRouter, HTTPException

from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.explainability_service import ExplainabilityService
from backend.services.model_service import ModelService
from backend.utils.text import clean_text, extract_keywords

router = APIRouter()
model_service = ModelService()
explainability_service = ExplainabilityService(model_service.model, model_service.tokenizer)


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    text = clean_text(request.text)
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    prediction_result = model_service.predict(text)
    important_keywords: list[str] = []

    try:
        keyword_scores = explainability_service.get_importance(text)
        important_keywords = extract_keywords([token for token, _ in keyword_scores], limit=8)
    except Exception:
        important_keywords = []

    return PredictionResponse(
        prediction=prediction_result["prediction"],
        confidence=prediction_result["confidence"],
        probabilities=prediction_result["probabilities"],
        important_keywords=important_keywords,
    )
