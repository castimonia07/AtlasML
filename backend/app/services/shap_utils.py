import os
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import pandas as pd

from app.core.config import REPORTS_DIR


def generate_shap_summary(model, X_sample, feature_names=None, model_kind: str = "tree") -> str | None:
    """Computes SHAP values for a sample of rows and saves a summary plot.
    Returns the file path, or None if it could not be computed."""
    try:
        # Convert X_sample to a DataFrame with feature_names if available
        if feature_names is not None and len(feature_names) == X_sample.shape[1]:
            X_df = pd.DataFrame(X_sample, columns=feature_names)
        else:
            X_df = X_sample

        if model_kind == "tree":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_df, min(50, len(X_df))))

        shap_values = explainer.shap_values(X_df)

        plt.figure()
        shap.summary_plot(shap_values, X_df, show=False)
        filename = f"shap_{uuid.uuid4().hex}.png"
        filepath = os.path.join(REPORTS_DIR, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=110, bbox_inches="tight")
        plt.close("all")
        return filepath
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None
