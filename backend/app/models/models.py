import datetime as dt

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, JSON, Float, Boolean
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    n_rows = Column(Integer, nullable=True)
    n_cols = Column(Integer, nullable=True)
    target_column = Column(String, nullable=True)
    date_column = Column(String, nullable=True)
    profile_json = Column(JSON, nullable=True)
    version = Column(Integer, default=1)
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)

    project = relationship("Project", back_populates="datasets")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    workflow_type = Column(String, nullable=False)   # supervised | unsupervised | time_series
    recommendation_reason = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, default="pending")        # pending | running | completed | failed
    mlflow_run_id = Column(String, nullable=True)
    best_model_name = Column(String, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    model_path = Column(String, nullable=True)
    shap_plot_path = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    
    # Advanced AutoML Fields
    model_version = Column(Integer, nullable=True)
    model_status = Column(String, default="candidate") # production | previous | candidate
    pipeline_logs = Column(Text, nullable=True)
    pipeline_progress = Column(String, default="pending") # pending | cleaning | engineering | training | evaluation | shap | completed | failed
    leaderboard_json = Column(JSON, nullable=True)
    hyperparameter_tuning = Column(Boolean, default=False)
    custom_hyperparameters = Column(JSON, nullable=True)
    business_objective = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="experiments")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    model_name = Column(String, nullable=True)
    input_data = Column(JSON, nullable=True)
    prediction = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
