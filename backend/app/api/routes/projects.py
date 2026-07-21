from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Project
from app.schemas.schemas import ProjectCreate, ProjectOut
from app.api.deps import get_owned_project

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("", response_model=ProjectOut)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
):
    project = Project(name=payload.name, description=payload.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.id.desc()).all()


@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    from app.models.models import Dataset, Experiment
    n_projects = db.query(Project).count()
    n_datasets = db.query(Dataset).count()
    n_experiments = db.query(Experiment).count()
    n_production = db.query(Experiment).filter(Experiment.model_status == "production").count()

    top_model = db.query(Experiment).filter(Experiment.model_status == "production").first()
    top_model_name = top_model.best_model_name if top_model else "N/A"

    return {
        "projects": n_projects,
        "datasets": n_datasets,
        "experiments": n_experiments,
        "production_models": n_production,
        "top_model": top_model_name
    }


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project=Depends(get_owned_project)):
    return project


@router.put("/{project_id}", response_model=ProjectOut)
def update_project(
    payload: ProjectCreate,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    project.name = payload.name
    project.description = payload.description
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}")
def delete_project(project=Depends(get_owned_project), db: Session = Depends(get_db)):
    db.delete(project)
    db.commit()
    return {"ok": True}
