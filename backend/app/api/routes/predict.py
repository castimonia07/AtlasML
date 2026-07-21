import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Experiment, Project, PredictionHistory
from app.schemas.schemas import PredictRequest
from app.api.deps import get_owned_project

router = APIRouter(prefix="/api/projects/{project_id}/experiments/{experiment_id}/predict", tags=["predict"])


@router.post("")
def predict(
    experiment_id: int,
    payload: PredictRequest,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    experiment = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id, Experiment.project_id == project.id)
        .first()
    )
    if not experiment or experiment.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment has no completed model")

    bundle = joblib.load(experiment.model_path)
    df = pd.DataFrame(payload.records)
    predictions_list = []

    if experiment.workflow_type == "supervised":
        X = df[bundle["feature_columns"]]
        Xt = bundle["preprocessor"].transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        preds = bundle["estimator"].predict(Xt)
        if "label_encoder" in bundle and bundle["label_encoder"] is not None:
            preds = bundle["label_encoder"].inverse_transform(preds)
        predictions_list = preds.tolist()

        # Calculate prediction probabilities for classification
        probabilities_list = None
        if hasattr(bundle["estimator"], "predict_proba"):
            try:
                probs = bundle["estimator"].predict_proba(Xt)
                probabilities_list = np.max(probs, axis=1).tolist()
            except Exception:
                pass
        result_payload = {"predictions": predictions_list, "probabilities": probabilities_list}

    elif experiment.workflow_type == "time_series":
        feat_cols = bundle.get("feature_columns") or bundle.get("feature_configuration") or []
        available_cols = [c for c in feat_cols if c in df.columns]
        X = df[available_cols] if available_cols else df
        
        estimator = bundle.get("estimator") or bundle.get("model")
        task = bundle.get("task")
        
        try:
            if task == "prophet":
                date_col = bundle.get("date_column") or "ds"
                if date_col in df.columns:
                    prophet_df = pd.DataFrame({"ds": pd.to_datetime(df[date_col])})
                else:
                    prophet_df = pd.DataFrame({"ds": pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq='D')})
                
                exog_cols = [c for c in feat_cols if c not in (bundle.get("target_column"), date_col, "ds")]
                exog_df = df[exog_cols] if exog_cols and all(c in df.columns for c in exog_cols) else None
                
                pred_df = estimator.predict(prophet_df, exog=exog_df)
                predictions_list = [float(v) for v in pred_df["yhat"].values]
            else:
                try:
                    preds = estimator.forecast(steps=len(df))
                except Exception:
                    try:
                        preds = estimator.predict(start=0, end=len(df)-1)
                    except Exception:
                        preds = estimator.predict(X)
                
                if hasattr(preds, "values"):
                    preds = preds.values
                predictions_list = [float(v) for v in preds]
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Failed to generate forecasting prediction: {err}")
            
        result_payload = {"predictions": predictions_list}

    elif experiment.workflow_type == "unsupervised":
        X = df[bundle["feature_columns"]]
        Xt = bundle["preprocessor"].transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        labels = bundle["kmeans"].predict(Xt)
        predictions_list = labels.tolist()
        result_payload = {"cluster": predictions_list}
    else:
        raise HTTPException(status_code=400, detail="Unsupported workflow type")

    # Log to prediction history database
    history = PredictionHistory(
        project_id=project.id,
        experiment_id=experiment_id,
        model_name=experiment.best_model_name,
        input_data=payload.records,
        prediction=predictions_list
    )
    db.add(history)
    db.commit()

    return result_payload


@router.get("/history")
def get_prediction_history(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    history = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.project_id == project.id, PredictionHistory.experiment_id == experiment_id)
        .order_by(PredictionHistory.timestamp.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": h.id,
            "model_name": h.model_name,
            "input_data": h.input_data,
            "prediction": h.prediction,
            "timestamp": h.timestamp
        }
        for h in history
    ]
