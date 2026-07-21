import os
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import DATASETS_DIR
from app.models.models import Dataset, Project
from app.schemas.schemas import DatasetOut, DatasetProfile, SetTargetRequest, RecommendationOut
from app.api.deps import get_owned_project
from app.services.profiling import profile_dataframe
from app.services.recommender import recommend_workflow

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["datasets"])


def _load_df(dataset: Dataset) -> pd.DataFrame:
    return pd.read_csv(dataset.filepath)


@router.post("", response_model=DatasetOut)
def upload_dataset(
    project: Project = Depends(get_owned_project),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    stored_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(DATASETS_DIR, stored_name)
    with open(filepath, "wb") as f:
        f.write(file.file.read())

    df = pd.read_csv(filepath)
    profile = profile_dataframe(df)

    # Calculate dataset version
    existing_count = db.query(Dataset).filter(Dataset.project_id == project.id).count()
    version = existing_count + 1

    dataset = Dataset(
        project_id=project.id,
        filename=file.filename,
        filepath=filepath,
        n_rows=profile["n_rows"],
        n_cols=profile["n_cols"],
        target_column=profile["suggested_target"],
        date_column=profile["detected_date_columns"][0] if profile["detected_date_columns"] else None,
        profile_json=profile,
        version=version,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("", response_model=list[DatasetOut])
def list_datasets(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    return db.query(Dataset).filter(Dataset.project_id == project.id).order_by(Dataset.id.desc()).all()


def _get_dataset(dataset_id: int, project: Project, db: Session) -> Dataset:
    ds = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.project_id == project.id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
def get_profile(
    dataset_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    ds = _get_dataset(dataset_id, project, db)
    return ds.profile_json


@router.post("/{dataset_id}/target", response_model=DatasetOut)
def set_target(
    dataset_id: int,
    payload: SetTargetRequest,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    ds = _get_dataset(dataset_id, project, db)
    ds.target_column = payload.target_column
    ds.date_column = payload.date_column
    
    # Re-run profiling with new target column configuration
    df = _load_df(ds)
    profile = profile_dataframe(df, ds.target_column)
    ds.profile_json = profile
    
    db.commit()
    db.refresh(ds)
    return ds


@router.get("/{dataset_id}/recommend", response_model=RecommendationOut)
def get_recommendation(
    dataset_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    ds = _get_dataset(dataset_id, project, db)
    df = _load_df(ds)
    rec = recommend_workflow(df, ds.target_column, ds.date_column, project.description)
    return rec


@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    ds = _get_dataset(dataset_id, project, db)
    if ds.filepath and os.path.exists(ds.filepath):
        try:
            os.remove(ds.filepath)
        except Exception:
            pass
    db.delete(ds)
    db.commit()
    return {"ok": True}
