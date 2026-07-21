import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.database import SessionLocal
from app.models.models import Experiment, Project, Dataset

db = SessionLocal()
try:
    print("--- PROJECTS ---")
    for p in db.query(Project).all():
        print(f"Project ID: {p.id}, Name: {p.name}")
        
    print("\n--- DATASETS ---")
    for d in db.query(Dataset).all():
        print(f"Dataset ID: {d.id}, Filename: {d.filename}, Path: {d.filepath}")
        
    print("\n--- EXPERIMENTS ---")
    for e in db.query(Experiment).all():
        print(f"Experiment ID: {e.id}, Project: {e.project_id}, Status: {e.status}, Model Name: {e.best_model_name}")
        print(f"  Model Path: {e.model_path}")
        print(f"  SHAP Path: {e.shap_plot_path}")
        print(f"  Model Exists: {os.path.exists(e.model_path) if e.model_path else False}")
finally:
    db.close()
