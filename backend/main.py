from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.routers.health import router as health_router
from backend.routers.info import router as info_router
from backend.routers.predict import router as predict_router
from backend.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Predict whether a news article is Fake or Real using a fine-tuned BERT model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(info_router)
app.include_router(predict_router)


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running."}


@app.on_event("startup")
def startup_event() -> None:
    logger.info("Starting Fake News Detector API")
