from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=4096, description="News article or headline text")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    important_keywords: list[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    framework: str
    dataset: str
    num_classes: int
    training_accuracy: float


class HealthResponse(BaseModel):
    status: str
    uptime: float
