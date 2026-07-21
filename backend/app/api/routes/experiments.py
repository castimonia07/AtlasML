import os
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd

from app.core.database import get_db
from app.models.models import Experiment, Dataset, Project
from app.schemas.schemas import ExperimentCreate, ExperimentOut, DriftRequest, ForecastMonitoringRequest
from app.api.deps import get_owned_project
from app.services.recommender import recommend_workflow
from app.services.orchestrator import run_experiment

router = APIRouter(prefix="/api/projects/{project_id}/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentOut)
def create_experiment(
    payload: ExperimentCreate,
    background_tasks: BackgroundTasks,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == payload.dataset_id, Dataset.project_id == project.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    workflow_type = payload.workflow_type
    reason, confidence = None, None
    if not workflow_type:
        df = pd.read_csv(dataset.filepath)
        rec = recommend_workflow(df, dataset.target_column, dataset.date_column, project.description)
        workflow_type, reason, confidence = rec["workflow_type"], rec["reason"], rec["confidence"]

    experiment = Experiment(
        project_id=project.id,
        dataset_id=dataset.id,
        workflow_type=workflow_type,
        recommendation_reason=reason,
        confidence=confidence,
        status="pending",
        hyperparameter_tuning=payload.hyperparameter_tuning or False,
        custom_hyperparameters=payload.custom_hyperparameters,
        business_objective=payload.business_objective,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    background_tasks.add_task(run_experiment, experiment.id)

    return experiment


@router.get("", response_model=list[ExperimentOut])
def list_experiments(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    return (
        db.query(Experiment)
        .filter(Experiment.project_id == project.id)
        .order_by(Experiment.id.desc())
        .all()
    )


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id, Experiment.project_id == project.id)
        .first()
    )
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.get("/{experiment_id}/download")
def download_model(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id, Experiment.project_id == project.id)
        .first()
    )
    if not exp or not exp.model_path or not os.path.exists(exp.model_path):
        raise HTTPException(status_code=404, detail="Model file not found or not ready yet")

    filename = f"model_project_{project.id}_v{exp.model_version or experiment_id}.joblib"
    return FileResponse(exp.model_path, media_type="application/octet-stream", filename=filename)


@router.post("/{experiment_id}/drift")
def detect_drift(
    experiment_id: int,
    payload: DriftRequest,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db)
):
    exp = _get_experiment(experiment_id, project, db)
    if exp.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment must be completed to detect drift")
        
    ref_dataset = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    if not ref_dataset:
        raise HTTPException(status_code=404, detail="Reference dataset not found")
        
    target_dataset = db.query(Dataset).filter(
        Dataset.id == payload.target_dataset_id,
        Dataset.project_id == project.id
    ).first()
    if not target_dataset:
        raise HTTPException(status_code=404, detail="Target dataset not found")

    try:
        ref_df = pd.read_csv(ref_dataset.filepath)
        target_df = pd.read_csv(target_dataset.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read datasets: {e}")

    from app.services.monitoring import calculate_drift
    result = calculate_drift(ref_df, target_df, exclude_cols=[ref_dataset.target_column, ref_dataset.date_column])
    return result


@router.post("/{experiment_id}/forecast-monitoring")
def forecast_monitoring(
    experiment_id: int,
    payload: ForecastMonitoringRequest,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db)
):
    exp = _get_experiment(experiment_id, project, db)
    if exp.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment must be completed to run forecast monitoring")
        
    ref_dataset = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    if not ref_dataset:
        raise HTTPException(status_code=404, detail="Reference dataset not found")
        
    target_dataset = db.query(Dataset).filter(
        Dataset.id == payload.target_dataset_id,
        Dataset.project_id == project.id
    ).first()
    if not target_dataset:
        raise HTTPException(status_code=404, detail="Target dataset not found")

    try:
        ref_df = pd.read_csv(ref_dataset.filepath)
        target_df = pd.read_csv(target_dataset.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read datasets: {e}")

    target_col = ref_dataset.target_column
    date_col = ref_dataset.date_column
    if not target_col or not date_col or target_col not in target_df.columns or date_col not in target_df.columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Target dataset must contain both the target column '{target_col}' and date column '{date_col}' used in training"
        )
        
    import joblib
    import numpy as np
    if not exp.model_path or not os.path.exists(exp.model_path):
         raise HTTPException(status_code=400, detail="Model file not found")
         
    try:
        model_data = joblib.load(exp.model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model file: {e}")
        
    from app.services.monitoring import calculate_drift
    from app.services.pipelines.timeseries import calculate_mape
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    drift_result = calculate_drift(ref_df, target_df, exclude_cols=[target_col, date_col])
    
    target_df[date_col] = pd.to_datetime(target_df[date_col], format="mixed")
    target_df = target_df.sort_values(date_col).reset_index(drop=True)
    y_actual = target_df[target_col].astype(float)
    
    estimator = model_data.get("estimator") or model_data.get("model")
    preds = []
    
    try:
        if hasattr(estimator, "predict"):
            task = model_data.get("task")
            if task == "prophet":
                input_df = pd.DataFrame({"ds": target_df[date_col]})
                pred_df = estimator.predict(input_df)
                preds = pred_df["yhat"].values
            else:
                pass
        
        if len(preds) == 0:
            if hasattr(estimator, "forecast"):
                preds = estimator.forecast(steps=len(y_actual))
                if hasattr(preds, "values"):
                    preds = preds.values
            elif hasattr(estimator, "predict"):
                preds = estimator.predict(start=0, end=len(y_actual)-1)
                if hasattr(preds, "values"):
                    preds = preds.values
                    
        if len(preds) == 0:
             raise ValueError("Model does not support forecast or predict methods")
             
        align_len = min(len(y_actual), len(preds))
        y_act_aligned = y_actual.iloc[:align_len]
        preds_aligned = preds[:align_len]
        
        rmse = float(np.sqrt(mean_squared_error(y_act_aligned, preds_aligned)))
        mae = float(mean_absolute_error(y_act_aligned, preds_aligned))
        mape = calculate_mape(y_act_aligned, preds_aligned)
        
        best_metrics = (exp.metrics_json or {}).get("best_metrics") or {}
        base_mape = best_metrics.get("mape", 0.0)
        
        degraded = False
        if base_mape > 0:
            degraded = (mape - base_mape) > 10.0 or (mape / base_mape) > 1.5
        else:
            degraded = mape > 20.0
            
        return {
            "drift_result": drift_result,
            "metrics": {
                "rmse": rmse,
                "mae": mae,
                "mape": mape
            },
            "baseline_metrics": best_metrics,
            "status": "Degraded" if degraded else "Normal",
            "summary": f"Forecast monitoring complete. Validation MAPE is {round(mape, 2)}% (Training Baseline: {round(base_mape, 2)}%). Model status is { 'DEGRADED' if degraded else 'NORMAL' }."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate predictions for monitoring: {e}")


@router.delete("/{experiment_id}")
def delete_experiment(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = _get_experiment(experiment_id, project, db)
    if exp.model_path and os.path.exists(exp.model_path):
        try:
            os.remove(exp.model_path)
        except Exception:
            pass
    if exp.shap_plot_path and os.path.exists(exp.shap_plot_path):
        try:
            os.remove(exp.shap_plot_path)
        except Exception:
            pass
    db.delete(exp)
    db.commit()
    return {"ok": True}


@router.post("/{experiment_id}/stop")
def stop_experiment(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = _get_experiment(experiment_id, project, db)
    if exp.status in ("pending", "running"):
        exp.status = "stopped"
        exp.pipeline_progress = "stopped"
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp.pipeline_logs = (exp.pipeline_logs or "") + f"[{ts}] STOPPED: Training execution interrupted by user.\n"
        exp.error = "Training stopped by user."
        db.commit()
    return exp


def _get_experiment(experiment_id: int, project: Project, db: Session) -> Experiment:
    exp = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id, Experiment.project_id == project.id)
        .first()
    )
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp

