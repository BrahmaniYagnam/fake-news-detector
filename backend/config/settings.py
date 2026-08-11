from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover
    from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Fake News Detector"
    app_version: str = "1.0.0"
    model_dir: Path = Path("saved_model")
    model_name: str = "fake-news-bert"
    log_level: str = "INFO"
    allowed_origins: list[str] = ["*"]
    max_length: int = 256
    batch_size: int = 16
    groq_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
