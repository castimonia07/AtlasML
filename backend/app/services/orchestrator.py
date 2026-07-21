import datetime as dt
import traceback

import pandas as pd

from app.core.database import SessionLocal
from app.models.models import Experiment, Dataset
from app.services.pipelines.supervised import run_supervised_pipeline
from app.services.pipelines.unsupervised import run_unsupervised_pipeline
from app.services.pipelines.timeseries import run_timeseries_pipeline


def run_experiment(experiment_id: int):
    """Runs in a FastAPI BackgroundTask, so it opens its own DB session
    rather than reusing the request-scoped one (which is already closed)."""
    db = SessionLocal()
    try:
        _run_experiment(experiment_id, db)
    finally:
        db.close()


def _run_experiment(experiment_id: int, db):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        return

    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()

    experiment.status = "running"
    db.commit()

    try:
        df = pd.read_csv(dataset.filepath)

        if experiment.workflow_type == "supervised":
            result = run_supervised_pipeline(
                df,
                dataset.target_column,
                experiment.id,
                hyperparameter_tuning=getattr(experiment, "hyperparameter_tuning", False),
                custom_hyperparameters=getattr(experiment, "custom_hyperparameters", None)
            )
        elif experiment.workflow_type == "unsupervised":
            result = run_unsupervised_pipeline(
                df,
                experiment.id,
                business_objective=getattr(experiment, "business_objective", None),
                custom_hyperparameters=getattr(experiment, "custom_hyperparameters", None)
            )
        elif experiment.workflow_type == "time_series":
            result = run_timeseries_pipeline(
                df,
                dataset.target_column,
                dataset.date_column,
                experiment.id,
                business_objective=getattr(experiment, "business_objective", None),
                custom_hyperparameters=getattr(experiment, "custom_hyperparameters", None),
                hyperparameter_tuning=getattr(experiment, "hyperparameter_tuning", False)
            )
        else:
            raise ValueError(f"Unknown workflow_type: {experiment.workflow_type}")

        experiment.status = "completed"
        experiment.mlflow_run_id = result["mlflow_run_id"]
        experiment.best_model_name = result["best_model_name"]
        experiment.metrics_json = result
        experiment.model_path = result["model_path"]
        experiment.shap_plot_path = result.get("shap_plot_path")
        experiment.leaderboard_json = result.get("leaderboard")
        experiment.completed_at = dt.datetime.utcnow()

        # Sequentially calculate version
        completed_count = db.query(Experiment).filter(
            Experiment.project_id == experiment.project_id,
            Experiment.status == "completed"
        ).count()
        experiment.model_version = completed_count + 1

        # Check production promotion
        prev_prod = db.query(Experiment).filter(
            Experiment.project_id == experiment.project_id,
            Experiment.model_status == "production"
        ).first()

        should_promote = False
        if not prev_prod:
            should_promote = True
        else:
            try:
                new_metrics = result.get("best_metrics") or result.get("metrics") or {}
                prev_metrics = (prev_prod.metrics_json or {}).get("best_metrics") or (prev_prod.metrics_json or {}).get("metrics") or {}
                
                if experiment.workflow_type == "supervised":
                    is_clf = result.get("task") == "classification"
                    metric_name = "accuracy" if is_clf else "r2"
                    new_val = new_metrics.get(metric_name, -999999)
                    prev_val = prev_metrics.get(metric_name, -999999)
                    if new_val > prev_val:
                        should_promote = True
                elif experiment.workflow_type == "unsupervised":
                    new_val = new_metrics.get("silhouette_score") or new_metrics.get("score") or -999999
                    prev_val = prev_metrics.get("silhouette_score") or prev_metrics.get("score") or -999999
                    if new_val > prev_val:
                        should_promote = True
                elif experiment.workflow_type == "time_series":
                    new_val = new_metrics.get("rmse") or new_metrics.get("mape") or 999999
                    prev_val = prev_metrics.get("rmse") or prev_metrics.get("mape") or 999999
                    if new_val < prev_val:
                        should_promote = True
            except Exception:
                should_promote = True

        if should_promote:
            if prev_prod:
                prev_prod.model_status = "previous"
            experiment.model_status = "production"
        else:
            experiment.model_status = "candidate"

    except Exception as e:
        db.refresh(experiment)
        if experiment.status != "stopped":
            experiment.status = "failed"
            experiment.pipeline_progress = "failed"
            experiment.error = f"{e}\n{traceback.format_exc()}"
        else:
            experiment.pipeline_progress = "stopped"

    db.commit()
