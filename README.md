---
title: Atlasml Backend
emoji: 📊
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# AtlasML

An end-to-end automated ML platform matching the architecture diagram: upload a
dataset, get a workflow recommendation, train and compare models, explain the
best one with SHAP, and serve predictions — all tracked in MLflow.

```
User → Next.js/Tailwind frontend → FastAPI backend
  → Auth (JWT) · Projects · Datasets · Reports
  → Profiling engine (schema check, stats, target/date detection)
  → Recommendation engine (supervised / unsupervised / time series)
  → Pipeline orchestrator → chosen pipeline
  → MLflow tracking → SHAP explainability → Model management
  → Prediction API → Dashboard (Plotly) → PostgreSQL
```

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | Next.js, React, Tailwind CSS, Plotly |
| Backend | FastAPI, SQLAlchemy, PostgreSQL, JWT |
| ML | Pandas, NumPy, Scikit-learn, TensorFlow*, XGBoost, LightGBM, CatBoost, SHAP, MLflow |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Docker, Docker Compose |
| Tools | Git, GitHub |

\* Scikit-learn / XGBoost / LightGBM / CatBoost handle the supervised, unsupervised
and time-series pipelines out of the box. TensorFlow is listed for future deep
model support and isn't wired into a pipeline yet — see "Extending" below.

## Run it with Docker (recommended)

```bash
git clone <this-repo>
cd AtlasML
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend docs (Swagger): http://localhost:8000/docs
- MLflow UI: http://localhost:5000

Register an account in the UI, create a project, upload a CSV, and walk
through the four steps: dataset → profile → recommendation → train.

## Run it locally without Docker

**Postgres**: have a local instance running, or `docker run -p 5432:5432 -e POSTGRES_USER=automl -e POSTGRES_PASSWORD=automl -e POSTGRES_DB=automl postgres:16-alpine`

**MLflow** (optional but recommended):
```bash
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000
```

**Backend**:
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adjust if needed
uvicorn app.main:app --reload
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```
Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local` if
the backend isn't on the default port.

## How the pipelines work

- **Profiling engine** (`app/services/profiling.py`) — infers dtypes, null
  counts, duplicate rows, numeric summaries, likely date columns, and a
  suggested target column.
- **Recommendation engine** (`app/services/recommender.py`) — rule-based:
  date + target → time series; no target → unsupervised; numeric target with
  many unique values → regression; otherwise → classification.
- **Pipeline orchestrator** (`app/services/orchestrator.py`) — runs as a
  FastAPI background task, dispatches to one of three pipelines, and writes
  status/results back onto the `Experiment` row so the frontend can poll it.
  - **Supervised** (`pipelines/supervised.py`) — trains logistic/linear
    regression, random forest, XGBoost, LightGBM and CatBoost, logs every run
    to MLflow (nested runs), and keeps the best model by accuracy/R².
  - **Unsupervised** (`pipelines/unsupervised.py`) — PCA for visualization
    plus KMeans with a silhouette-score sweep over k.
  - **Time series** (`pipelines/timeseries.py`) — lag-feature forecasting
    with a random forest, evaluated on a chronological hold-out split.
- **Explainability** (`app/services/shap_utils.py`) — SHAP `TreeExplainer`
  summary plot for the winning supervised model, saved as PNG and served by
  the API.
- **Model management** — the winning model (plus its preprocessing pipeline)
  is serialized with `joblib` and referenced from the `Experiment` row; the
  prediction API loads it back for inference.
- **Reports** (`app/api/routes/reports.py`) — generates a PDF summary of an
  experiment (metrics + SHAP plot) with `fpdf2`.

## Project layout

```
backend/
  app/
    core/         settings, DB session, JWT/password hashing
    models/       SQLAlchemy models (User, Project, Dataset, Experiment)
    schemas/      Pydantic request/response models
    api/routes/   auth, projects, datasets, experiments, predict, reports
    services/     profiling, recommender, orchestrator, mlflow/shap helpers
    services/pipelines/   supervised, unsupervised, time_series
frontend/
  app/            Next.js App Router pages (login, register, dashboard, project detail)
  components/     Navbar, PlotlyChart
  lib/api.ts      axios client + shared types
docker-compose.yml
```

## Extending

- Swap the rule-based recommender for a learned meta-model once you have
  logged enough experiments in MLflow to train on.
- Add a TensorFlow/Keras model into `pipelines/supervised.py`'s candidate
  dict for deep tabular baselines.
- Move `BackgroundTasks` to a real queue (Celery/RQ) if training needs to
  scale beyond a single backend process.
