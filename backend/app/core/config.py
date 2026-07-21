import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AtlasML"

    STORAGE_DIR: str = os.getenv(
        "STORAGE_DIR", os.path.join(os.path.dirname(__file__), "..", "storage")
    )

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'storage', 'automl.db')).replace('\\', '/')}",
    )

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    CORS_ORIGINS: list = [
        x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()
    ] if os.getenv("CORS_ORIGINS") else [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://frontend:3000"
    ]

    class Config:
        env_file = ".env"


settings = Settings()

DATASETS_DIR = os.path.join(settings.STORAGE_DIR, "datasets")
MODELS_DIR = os.path.join(settings.STORAGE_DIR, "models")
REPORTS_DIR = os.path.join(settings.STORAGE_DIR, "reports")

for d in [DATASETS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
