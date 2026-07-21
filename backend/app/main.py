from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import Base, engine
from app.models import models  # noqa: F401 (ensures models are registered)
from app.api.routes import projects, datasets, experiments, predict, reports

from sqlalchemy import inspect

# Self-healing database schema check (for local SQLite dev)
try:
    inspector = inspect(engine)
    if "experiments" in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns("experiments")]
        if "leaderboard_json" not in cols:
            Base.metadata.drop_all(bind=engine)
        else:
            from sqlalchemy import text
            with engine.begin() as conn:
                if "hyperparameter_tuning" not in cols:
                    conn.execute(text("ALTER TABLE experiments ADD COLUMN hyperparameter_tuning BOOLEAN DEFAULT 0"))
                if "custom_hyperparameters" not in cols:
                    conn.execute(text("ALTER TABLE experiments ADD COLUMN custom_hyperparameters JSON"))
                if "business_objective" not in cols:
                    conn.execute(text("ALTER TABLE experiments ADD COLUMN business_objective VARCHAR(255)"))
except Exception as e:
    print(f"Migration error: {e}")

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(predict.router)
app.include_router(reports.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
