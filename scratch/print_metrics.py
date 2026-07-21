import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.database import SessionLocal
from app.models.models import Experiment

db = SessionLocal()
try:
    exps = db.query(Experiment).filter(Experiment.status == "completed").all()
    for e in exps:
        print(f"ID: {e.id}, Model: {e.best_model_name}")
        print("  metrics_json:", e.metrics_json)
        print("  leaderboard_json:", e.leaderboard_json)
        print("-" * 50)
finally:
    db.close()
