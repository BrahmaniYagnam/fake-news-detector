from datetime import datetime

from fastapi import APIRouter

from backend.schemas.prediction import HealthResponse

router = APIRouter()
start_time = datetime.utcnow()


@router.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check() -> HealthResponse:
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return HealthResponse(status="ok", uptime=uptime)
