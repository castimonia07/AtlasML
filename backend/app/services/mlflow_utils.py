import os
import mlflow
import requests

from app.core.config import settings


def init_mlflow(experiment_name: str = "automl-platform"):
    uri = settings.MLFLOW_TRACKING_URI
    is_up = True
    
    # Quick health check if tracking URI is an HTTP/HTTPS endpoint
    if uri.startswith("http"):
        try:
            # We set a very short timeout (0.5s) to avoid blocking the pipeline training
            requests.get(uri, timeout=0.5)
        except Exception:
            is_up = False

    if is_up:
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
        except Exception:
            is_up = False

    if not is_up:
        print(f"MLflow server at {uri} is unreachable. Falling back to local file-based tracking...")
        local_mlflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "mlflow"))
        os.makedirs(local_mlflow_dir, exist_ok=True)
        local_uri_path = local_mlflow_dir.replace('\\', '/')
        local_uri = f"file:///{local_uri_path}"
        mlflow.set_tracking_uri(local_uri)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Failed to set local MLflow experiment: {e}")
