import os
import time
import uuid
import datetime as dt

import joblib
import mlflow
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score

from app.core.config import MODELS_DIR, REPORTS_DIR
from app.core.database import SessionLocal
from app.models.models import Experiment
from app.ml.preprocessing import split_feature_types, build_preprocessor
from app.services.mlflow_utils import init_mlflow


def update_progress(experiment_id: int, progress: str, log_msg: str):
    db = SessionLocal()
    should_raise = False
    try:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            if exp.status in ("stopped", "failed"):
                should_raise = True
            else:
                exp.pipeline_progress = progress
                ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_logs = exp.pipeline_logs or ""
                exp.pipeline_logs = current_logs + f"[{ts}] {progress.upper()}: {log_msg}\n"
                db.commit()
    except Exception as e:
        print(f"Error updating progress: {e}")
    finally:
        db.close()
    if should_raise:
        raise ValueError("Training stopped by user.")


def describe_clusters(df: pd.DataFrame, labels: np.ndarray, numeric_cols: list, categorical_cols: list) -> tuple[dict, dict]:
    """Calculates sizes, feature averages, and categorical common traits for each cluster."""
    df_temp = df.copy()
    df_temp["__cluster"] = labels
    
    stats = {}
    descriptions = {}
    global_means = df[numeric_cols].mean() if numeric_cols else pd.Series()
    
    unique_labels = sorted(list(set(labels)))
    for clus in unique_labels:
        sub = df_temp[df_temp["__cluster"] == clus]
        size = len(sub)
        pct = (size / len(df)) * 100
        
        clus_means = sub[numeric_cols].mean() if numeric_cols else pd.Series()
        
        # Most common categories
        common_cats = {}
        for col in categorical_cols:
            if not sub[col].dropna().empty:
                val = sub[col].mode().iloc[0]
                common_cats[col] = str(val)
                
        stats[str(clus)] = {
            "size": size,
            "percentage": round(pct, 2),
            "means": {k: float(v) for k, v in clus_means.items() if not pd.isna(v)},
            "common_categories": common_cats
        }
        
        # Compare cluster mean vs global mean to build description
        traits = []
        for col in numeric_cols:
            g_val = global_means[col]
            c_val = clus_means[col]
            if g_val != 0 and not pd.isna(g_val) and not pd.isna(c_val):
                ratio = c_val / g_val
                if ratio > 1.25:
                    traits.append(f"high {col}")
                elif ratio < 0.75:
                    traits.append(f"low {col}")
                    
        # Add categorical traits
        for col, val in common_cats.items():
            traits.append(f"{col}={val}")
            
        if traits:
            descriptions[str(clus)] = f"Characterized by {', '.join(traits[:4])}."
        else:
            descriptions[str(clus)] = "Balanced group matching the overall dataset average."
            
    return stats, descriptions


def describe_anomalies(df: pd.DataFrame, is_anomaly: np.ndarray, numeric_cols: list) -> dict:
    """Calculates outlier ratio, and feature differences between normal and anomaly cases."""
    df_temp = df.copy()
    df_temp["__is_anomaly"] = is_anomaly  # -1 for anomaly, 1 for normal
    
    normal_sub = df_temp[df_temp["__is_anomaly"] == 1]
    anomaly_sub = df_temp[df_temp["__is_anomaly"] == -1]
    
    total = len(df)
    anom_count = len(anomaly_sub)
    anom_pct = (anom_count / total) * 100
    
    normal_means = normal_sub[numeric_cols].mean() if numeric_cols else pd.Series()
    anomaly_means = anomaly_sub[numeric_cols].mean() if numeric_cols else pd.Series()
    
    key_differences = []
    for col in numeric_cols:
        norm_v = normal_means.get(col, 0)
        anom_v = anomaly_means.get(col, 0)
        if norm_v != 0 and not pd.isna(norm_v) and not pd.isna(anom_v):
            diff_ratio = abs(anom_v - norm_v) / abs(norm_v)
            if diff_ratio > 0.25:
                direction = "higher" if anom_v > norm_v else "lower"
                key_differences.append(f"Anomalies show {direction} {col} (avg: {round(anom_v, 2)} vs normal: {round(norm_v, 2)})")
                
    return {
        "anomaly_count": anom_count,
        "anomaly_percentage": round(anom_pct, 2),
        "key_observations": key_differences[:3]
    }


def generate_unsupervised_plots(Xp: np.ndarray, labels: np.ndarray, task: str, explained_variance_ratio: list = None) -> str:
    """Generates task-specific matplotlib subplots and saves a single PNG image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if task == "clustering":
        # PCA scatter plot
        scatter = axes[0].scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap="viridis", alpha=0.7, edgecolors="none", s=25)
        axes[0].set_title("PCA Cluster Scatter Plot")
        axes[0].set_xlabel("Component 1")
        axes[0].set_ylabel("Component 2")
        fig.colorbar(scatter, ax=axes[0], label="Cluster")
        
        # Cluster distribution bar plot
        unique, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100
        axes[1].bar([f"Cluster {int(u)}" for u in unique], percentages, color="#3B6D5C")
        axes[1].set_title("Cluster Size Distribution (%)")
        axes[1].set_ylabel("Percentage (%)")
        
    elif task == "anomaly_detection":
        # PCA scatter plot colored by anomaly status
        colors = ["#d9534f" if l == -1 else "#337ab7" for l in labels]
        axes[0].scatter(Xp[:, 0], Xp[:, 1], c=colors, alpha=0.7, edgecolors="none", s=25)
        axes[0].set_title("PCA Anomaly Scatter Plot")
        axes[0].set_xlabel("Component 1")
        axes[0].set_ylabel("Component 2")
        
        # Draw legend custom handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d9534f", label="Anomaly"),
            Patch(facecolor="#337ab7", label="Normal")
        ]
        axes[0].legend(handles=legend_elements)
        
        # Anomaly ratio bar plot
        unique, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100
        labels_str = ["Anomaly" if u == -1 else "Normal" for u in unique]
        axes[1].bar(labels_str, percentages, color=["#d9534f" if u == -1 else "#337ab7" for u in unique])
        axes[1].set_title("Anomaly Distribution (%)")
        axes[1].set_ylabel("Percentage (%)")
        
    elif task == "dimensionality_reduction" and explained_variance_ratio is not None:
        # Cumulative Explained Variance plot
        cum_var = np.cumsum(explained_variance_ratio) * 100
        axes[0].plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#3B6D5C", linestyle="--")
        axes[0].set_title("Cumulative Explained Variance")
        axes[0].set_xlabel("Number of Components")
        axes[0].set_ylabel("Variance Explained (%)")
        axes[0].set_ylim(0, 105)
        
        # Individual Explained Variance ratio bar plot
        axes[1].bar([f"PC {i+1}" for i in range(len(explained_variance_ratio))], np.array(explained_variance_ratio) * 100, color="#6A5ACD")
        axes[1].set_title("Variance contribution by PC (%)")
        axes[1].set_ylabel("Percentage (%)")
        
    plt.tight_layout()
    filename = f"unsupervised_{uuid.uuid4().hex}.png"
    filepath = os.path.join(REPORTS_DIR, filename)
    plt.savefig(filepath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return filepath


def run_unsupervised_pipeline(
    df: pd.DataFrame,
    experiment_id: int,
    business_objective: str = None,
    custom_hyperparameters: dict = None
) -> dict:
    init_mlflow()
    
    # 1. Map business need to actual task type
    task = "clustering"  # Default fallback
    if business_objective in ("fraud_detection", "outlier_detection"):
        task = "anomaly_detection"
    elif business_objective == "feature_reduction":
        task = "dimensionality_reduction"
        
    update_progress(experiment_id, "cleaning", f"Starting {task.upper()} pipeline cleaning. Dropping invalid records.")
    numeric_cols, categorical_cols = split_feature_types(df, exclude=[])
    
    # Clean target columns if they exist in dataset but we are running unsupervised
    X = df[numeric_cols + categorical_cols]
    
    update_progress(experiment_id, "engineering", "Scaling numeric fields and converting categorical fields.")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    update_progress(experiment_id, "training", f"Preprocessing dataset inputs.")
    with mlflow.start_run(run_name=f"experiment-{experiment_id}") as run:
        mlflow.log_param("workflow_type", "unsupervised")
        mlflow.log_param("task", task)
        mlflow.log_param("business_objective", str(business_objective))
        mlflow.log_param("n_rows", len(df))
        
        t_pre = time.time()
        Xt = preprocessor.fit_transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
            
        pre_time = time.time() - t_pre
        
        # PCA projection always created for visualization / 2D scatter plots
        n_components_pca = min(2, Xt.shape[1])
        pca = PCA(n_components=n_components_pca, random_state=42)
        Xp = pca.fit_transform(Xt)
        
        best_model = None
        best_labels = None
        best_score = -1.0
        best_algorithm = ""
        leaderboard = []
        stats = {}
        descriptions = {}
        insights = []
        anomaly_summary = {}
        explained_variance_ratio = []
        best_params = {}
        
        # Check if Advanced Mode is used (custom_hyperparameters has selected algorithm)
        is_advanced = custom_hyperparameters and "algorithm" in custom_hyperparameters
        
        # ----------------- CLUSTERING WORKFLOW -----------------
        if task == "clustering":
            if is_advanced:
                alg = custom_hyperparameters["algorithm"]
                update_progress(experiment_id, "training", f"Training user-selected {alg} model (Advanced Mode).")
                
                t_start = time.time()
                if alg == "kmeans":
                    n_clus = int(custom_hyperparameters.get("n_clusters", 3))
                    model = KMeans(n_clusters=n_clus, random_state=42, n_init=10)
                    labels = model.fit_predict(Xt)
                    best_params = {"n_clusters": n_clus}
                elif alg == "agglomerative":
                    n_clus = int(custom_hyperparameters.get("n_clusters", 3))
                    model = AgglomerativeClustering(n_clusters=n_clus)
                    labels = model.fit_predict(Xt)
                    best_params = {"n_clusters": n_clus}
                elif alg == "dbscan":
                    eps_val = float(custom_hyperparameters.get("eps", 0.5))
                    min_s = int(custom_hyperparameters.get("min_samples", 5))
                    model = DBSCAN(eps=eps_val, min_samples=min_s)
                    labels = model.fit_predict(Xt)
                    best_params = {"eps": eps_val, "min_samples": min_s}
                else:
                    raise ValueError(f"Unknown custom clustering algorithm: {alg}")
                    
                fit_time = time.time() - t_start
                score = float(silhouette_score(Xt, labels)) if len(set(labels)) > 1 else 0.0
                
                best_model = model
                best_labels = labels
                best_score = score
                best_algorithm = alg
                
                leaderboard.append({
                    "rank": 1,
                    "model_name": f"{alg}_advanced",
                    "metrics": {"silhouette_score": score},
                    "train_time": round(fit_time + pre_time, 4),
                    "inference_time": 0.01
                })
            else:
                # AUTO MODE: Recommend and sweep parameters
                update_progress(experiment_id, "training", "Sweeping clustering algorithms and sizes (Auto Mode).")
                candidate_runs = []
                max_k = min(8, len(df))
                
                # 1. KMeans Sweep
                if len(df) > 2 and max_k > 2:
                    for k in range(2, max_k):
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(Xt)
                        if len(set(labels)) > 1:
                            score = float(silhouette_score(Xt, labels))
                            candidate_runs.append(("kmeans", km, labels, score, {"n_clusters": k}))
                            
                # 2. Agglomerative Sweep
                if len(df) > 2 and max_k > 2:
                    for k in range(2, max_k):
                        agg = AgglomerativeClustering(n_clusters=k)
                        labels = agg.fit_predict(Xt)
                        if len(set(labels)) > 1:
                            score = float(silhouette_score(Xt, labels))
                            candidate_runs.append(("agglomerative", agg, labels, score, {"n_clusters": k}))
                            
                # 3. DBSCAN Sweep
                eps_options = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
                min_samples_val = max(3, 2 * len(numeric_cols))
                for eps_opt in eps_options:
                    dbscan_m = DBSCAN(eps=eps_opt, min_samples=min_samples_val)
                    labels = dbscan_m.fit_predict(Xt)
                    unique_labels = set(labels) - {-1} # exclude noise points
                    if len(unique_labels) > 1:
                        # calculate silhouette score on non-noise points
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 10:
                            score = float(silhouette_score(Xt[non_noise_mask], labels[non_noise_mask]))
                            candidate_runs.append(("dbscan", dbscan_m, labels, score, {"eps": eps_opt, "min_samples": min_samples_val}))
                            
                # Fallback if no runs succeeded
                if not candidate_runs:
                    fallback_k = 2 if len(df) >= 2 else 1
                    model = KMeans(n_clusters=fallback_k, random_state=42, n_init=10)
                    labels = model.fit_predict(Xt)
                    candidate_runs.append(("kmeans", model, labels, 0.0, {"n_clusters": fallback_k}))
                    
                # Rank and select champion
                candidate_runs = sorted(candidate_runs, key=lambda x: x[3], reverse=True)
                best_run = candidate_runs[0]
                best_algorithm, best_model, best_labels, best_score, best_params = best_run
                
                # Build leaderboard
                for i, run_item in enumerate(candidate_runs[:5]):
                    leaderboard.append({
                        "rank": i + 1,
                        "model_name": f"{run_item[0]}_{str(run_item[4])}",
                        "metrics": {"silhouette_score": run_item[3]},
                        "train_time": 0.05,
                        "inference_time": 0.005
                    })
                    
            # Set recommendation details
            rec_reason = f"Automatically recommended {best_algorithm.upper()} with parameters {best_params} because it achieved the highest Silhouette Score of {round(best_score, 4)} on preprocessed features."
            insights.append(f"Optimization: Segmented target items into {len(set(best_labels) - {-1})} clusters.")
            
            # Interpretability
            stats, descriptions = describe_clusters(df, best_labels, numeric_cols, categorical_cols)
            
            # Extract traits to insights
            for clusId, desc in descriptions.items():
                if clusId != "-1":
                    insights.append(f"Segment {clusId} Trait: {desc.replace('Characterized by ', '')}")
                    
        # ----------------- ANOMALY DETECTION WORKFLOW -----------------
        elif task == "anomaly_detection":
            contamination_val = 0.05  # Default
            if is_advanced:
                alg = custom_hyperparameters["algorithm"]
                contamination_val = float(custom_hyperparameters.get("contamination", 0.05))
            else:
                best_algorithm = "isolation_forest"
                
            best_algorithm = "isolation_forest"
            best_params = {"contamination": contamination_val}
            update_progress(experiment_id, "training", f"Training Isolation Forest model (contamination={contamination_val}).")
            
            t_start = time.time()
            model = IsolationForest(contamination=contamination_val, random_state=42)
            labels = model.fit_predict(Xt)  # returns -1 for anomaly, 1 for normal
            fit_time = time.time() - t_start
            
            best_model = model
            best_labels = labels
            
            # If target column exists, evaluate precision/recall
            # Try to check if target exists in dataset
            target_col = None
            db = SessionLocal()
            exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                dataset_obj = db.query(Experiment).filter(Experiment.id == experiment_id).first().project.datasets
                # Let's find dataset
                from app.models.models import Dataset
                ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
                if ds and ds.target_column and ds.target_column in df.columns:
                    target_col = ds.target_column
            db.close()
            
            eval_metrics = {}
            if target_col:
                y_true = df[target_col].dropna()
                # Encode binary anomaly ground truth: assume minority class or values containing 'anomaly'/'fraud'/1 are anomalies
                # Map true anomalies to -1 and normal to 1
                if not y_true.empty:
                    y_true_binary = np.where(y_true.astype(str).str.lower().str.contains("fraud|anomaly|1|yes"), -1, 1)
                    # Align shapes
                    align_len = min(len(y_true_binary), len(labels))
                    if align_len > 10:
                        prec = precision_score(y_true_binary[:align_len], labels[:align_len], pos_label=-1, zero_division=0)
                        rec = recall_score(y_true_binary[:align_len], labels[:align_len], pos_label=-1, zero_division=0)
                        f1 = f1_score(y_true_binary[:align_len], labels[:align_len], pos_label=-1, zero_division=0)
                        eval_metrics = {
                            "precision": float(prec),
                            "recall": float(rec),
                            "f1_score": float(f1)
                        }
                        
            anomaly_summary = describe_anomalies(df, best_labels, numeric_cols)
            best_score = float(anomaly_summary["anomaly_percentage"])
            eval_metrics["anomaly_ratio"] = best_score / 100.0
            
            leaderboard.append({
                "rank": 1,
                "model_name": f"isolation_forest_c{contamination_val}",
                "metrics": eval_metrics,
                "train_time": round(fit_time + pre_time, 4),
                "inference_time": 0.008
            })
            
            rec_reason = f"Automatically recommended Isolation Forest (contamination={contamination_val}) because it isolates multi-dimensional outliers efficiently using binary splits."
            insights.append(f"Fraud/Anomaly Ratio: Found {anomaly_summary['anomaly_count']} suspicious records ({anomaly_summary['anomaly_percentage']}% of dataset).")
            for obs in anomaly_summary["key_observations"]:
                insights.append(f"Observation: {obs}")
                
        # ----------------- DIMENSIONALITY REDUCTION WORKFLOW -----------------
        elif task == "dimensionality_reduction":
            best_algorithm = "pca"
            n_comp = min(3, Xt.shape[1])
            if is_advanced:
                n_comp = int(custom_hyperparameters.get("n_components", 2))
                
            n_comp = min(n_comp, Xt.shape[1])
            best_params = {"n_components": n_comp}
            update_progress(experiment_id, "training", f"Running Principal Component Analysis (components={n_comp}).")
            
            t_start = time.time()
            model = PCA(n_components=n_comp, random_state=42)
            model.fit(Xt)
            fit_time = time.time() - t_start
            
            best_model = model
            best_labels = np.zeros(len(df)) # dummy labels for PCA
            explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]
            best_score = float(sum(explained_variance_ratio))
            
            leaderboard.append({
                "rank": 1,
                "model_name": f"pca_c{n_comp}",
                "metrics": {"total_explained_variance": best_score},
                "train_time": round(fit_time + pre_time, 4),
                "inference_time": 0.001
            })
            
            rec_reason = f"PCA was selected to project variables into {n_comp} orthogonal features. These dimensions explain {round(best_score * 100, 2)}% of the total dataset variance."
            insights.append(f"Explained Variance: The principal components explain {round(best_score * 100, 2)}% of information.")
            
            # Analyze loadings to see top features contributing to PC1
            loadings = np.abs(model.components_[0])
            feature_names_in = numeric_cols + categorical_cols # approximation of preprocessed names
            try:
                raw_names = preprocessor.get_feature_names_out()
                feature_names_in = [n.split("__")[-1] for n in raw_names]
            except Exception:
                pass
                
            if len(loadings) == len(feature_names_in):
                top_contributors = [feature_names_in[idx] for idx in np.argsort(loadings)[::-1][:3]]
                insights.append(f"Top Drivers: Features '{', '.join(top_contributors)}' represent the highest contributors to the primary component variance.")
                
        # ----------------- VISUALIZATIONS GENERATION -----------------
        update_progress(experiment_id, "evaluation", "Compiling results and generating chart visualizations.")
        plot_path = generate_unsupervised_plots(
            Xp,
            best_labels,
            task,
            explained_variance_ratio=explained_variance_ratio if task == "dimensionality_reduction" else None
        )
        
        # Log to MLflow
        mlflow.log_metric("silhouette_score" if task == "clustering" else "explained_variance", float(best_score))
        mlflow.log_param("algorithm", best_algorithm)
        mlflow.log_param("best_params", str(best_params))
        
        # Save model joblib artifact
        model_id = uuid.uuid4().hex
        model_path = os.path.join(MODELS_DIR, f"model_{model_id}.joblib")
        joblib.dump({
            "model": best_model,
            "preprocessor": preprocessor,
            "pca_viz": pca,
            "feature_columns": numeric_cols + categorical_cols,
            "hyperparameters": best_params,
            "metrics": {"best_score": best_score},
            "timestamp": dt.datetime.now().isoformat(),
            "task": task,
            "algorithm": best_algorithm
        }, model_path)
        
        update_progress(experiment_id, "completed", f"Unsupervised {task} completed successfully.")
        
        return {
            "mlflow_run_id": run.info.run_id,
            "best_model_name": f"{best_algorithm}_{str(best_params)}",
            "metrics": {"best_score": best_score},
            "best_metrics": {"silhouette_score" if task == "clustering" else "score": float(best_score)},
            "leaderboard": leaderboard,
            "cluster_statistics": stats,
            "cluster_descriptions": descriptions,
            "anomaly_summary": anomaly_summary,
            "explained_variance_ratio": explained_variance_ratio,
            "business_insights": insights,
            "model_path": model_path,
            "shap_plot_path": plot_path,
            "task": task,
            "pca_points": Xp[:, :2].tolist(),
            "cluster_labels": [int(l) for l in best_labels],
            "recommendation_reason": rec_reason
        }
