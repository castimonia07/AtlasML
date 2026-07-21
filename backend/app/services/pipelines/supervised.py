import os
import time
import uuid
import datetime as dt

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from app.core.config import MODELS_DIR
from app.core.database import SessionLocal
from app.models.models import Experiment
from app.ml.preprocessing import split_feature_types, build_preprocessor
from app.services.mlflow_utils import init_mlflow
from app.services.shap_utils import generate_shap_summary


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


def _is_classification(y: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return False
    return True


def _candidate_models(is_clf: bool, random_state: int = 42) -> dict:
    if is_clf:
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
            "xgboost": XGBClassifier(
                n_estimators=200, eval_metric="logloss", random_state=random_state, verbosity=0
            ),
            "lightgbm": LGBMClassifier(n_estimators=200, random_state=random_state, verbose=-1),
            "catboost": CatBoostClassifier(
                iterations=200, random_state=random_state, verbose=False
            ),
        }
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "xgboost": XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0),
        "lightgbm": LGBMRegressor(n_estimators=200, random_state=random_state, verbose=-1),
        "catboost": CatBoostRegressor(iterations=200, random_state=random_state, verbose=False),
    }


def calculate_classification_plots(y_true, y_pred, y_prob=None, is_binary=True):
    """Calculates confusion matrix, ROC curve, and PR curve lists for UI visualization."""
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    fpr_list, tpr_list, roc_auc = [], [], 0.0
    if y_prob is not None:
        try:
            if is_binary:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    probs_pos = y_prob[:, 1]
                else:
                    probs_pos = y_prob
                fpr, tpr, _ = roc_curve(y_true, probs_pos)
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
                roc_auc = float(auc(fpr, tpr))
            else:
                # Multiclass ROC fallback: calculate class 0 vs rest
                fpr, tpr, _ = roc_curve(y_true == 0, y_prob[:, 0])
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
                roc_auc = float(auc(fpr, tpr))
        except Exception:
            pass

    precision_list, recall_list = [], []
    if y_prob is not None:
        try:
            if is_binary:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    probs_pos = y_prob[:, 1]
                else:
                    probs_pos = y_prob
                prec, rec, _ = precision_recall_curve(y_true, probs_pos)
                precision_list = prec.tolist()
                recall_list = rec.tolist()
            else:
                prec, rec, _ = precision_recall_curve(y_true == 0, y_prob[:, 0])
                precision_list = prec.tolist()
                recall_list = rec.tolist()
        except Exception:
            pass

    return {
        "confusion_matrix": cm_list,
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list, "auc": roc_auc},
        "precision_recall_curve": {"precision": precision_list, "recall": recall_list}
    }


def run_supervised_pipeline(
    df: pd.DataFrame,
    target_column: str,
    experiment_id: int,
    hyperparameter_tuning: bool = False,
    custom_hyperparameters: dict = None
) -> dict:
    init_mlflow()

    update_progress(experiment_id, "cleaning", "Starting data cleaning and dropping null targets.")
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    y = df[target_column]
    is_clf = _is_classification(y)
    label_encoder = None
    if is_clf:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    update_progress(experiment_id, "engineering", "Splitting feature columns and applying standard transformations.")
    numeric_cols, categorical_cols = split_feature_types(df, exclude=[target_column])
    X = df[numeric_cols + categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_clf and y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = _candidate_models(is_clf)

    # Fit preprocessor to get clean feature names out
    preprocessor.fit(X_train)
    try:
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [n.split("__")[-1] for n in raw_names]
    except Exception:
        feature_names = list(X_train.columns)

    results = []
    
    update_progress(experiment_id, "training", f"Entering model training phase. Training {len(models)} candidate models.")
    with mlflow.start_run(run_name=f"experiment-{experiment_id}") as parent_run:
        mlflow.log_param("workflow_type", "supervised")
        mlflow.log_param("task", "classification" if is_clf else "regression")
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_features", len(numeric_cols) + len(categorical_cols))
        mlflow.log_param("hyperparameter_tuning", str(hyperparameter_tuning))

        for name, estimator in models.items():
            update_progress(experiment_id, "training", f"Fitting {name} estimator on training fold.")
            with mlflow.start_run(run_name=name, nested=True):
                Xt_train = preprocessor.transform(X_train)
                Xt_test = preprocessor.transform(X_test)
                if hasattr(Xt_train, "toarray"):
                    Xt_train = Xt_train.toarray()
                    Xt_test = Xt_test.toarray()

                # Get param grid for current model
                param_grid = None
                if hyperparameter_tuning:
                    # check if user passed custom parameters for this model
                    if custom_hyperparameters and name in custom_hyperparameters:
                        param_grid = custom_hyperparameters[name]
                        update_progress(experiment_id, "training", f"Using custom hyperparameters grid for {name}.")
                    else:
                        # default parameter grids
                        if name == "logistic_regression":
                            param_grid = {
                                "C": [0.01, 0.1, 1.0, 10.0],
                                "penalty": ["l2"]
                            }
                        elif name == "linear_regression":
                            param_grid = {
                                "fit_intercept": [True, False]
                            }
                        elif name == "random_forest":
                            param_grid = {
                                "n_estimators": [50, 100, 200],
                                "max_depth": [5, 10, None],
                                "min_samples_split": [2, 5]
                            }
                        elif name == "xgboost":
                            param_grid = {
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                                "max_depth": [3, 5, 7]
                            }
                        elif name == "lightgbm":
                            param_grid = {
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.01, 0.05, 0.1],
                                "num_leaves": [15, 31, 63]
                            }
                        elif name == "catboost":
                            param_grid = {
                                "iterations": [100, 200],
                                "learning_rate": [0.01, 0.05, 0.1],
                                "depth": [4, 6, 8]
                            }

                # Measure training time
                t_start = time.time()
                if param_grid:
                    scoring_metric = "f1_weighted" if is_clf else "r2"
                    # Handle small datasets or edge cases by setting cv=min(3, length)
                    cv_folds = min(3, len(y_train))
                    if cv_folds < 2:
                        estimator.fit(Xt_train, y_train)
                    else:
                        try:
                            n_iter_budget = 5
                            if isinstance(param_grid, dict):
                                lengths = [len(v) for v in param_grid.values() if isinstance(v, (list, np.ndarray))]
                                if lengths:
                                    n_iter_budget = min(5, int(np.prod(lengths)))
                            
                            search = RandomizedSearchCV(
                                estimator,
                                param_distributions=param_grid,
                                n_iter=n_iter_budget,
                                cv=cv_folds,
                                random_state=42,
                                n_jobs=1,
                                scoring=scoring_metric
                            )
                            search.fit(Xt_train, y_train)
                            estimator = search.best_estimator_
                            # Log best params to mlflow
                            for pk, pv in search.best_params_.items():
                                mlflow.log_param(f"best_{pk}", pv)
                        except Exception as err:
                            print(f"Error during hyperparameter tuning for {name}: {err}")
                            update_progress(experiment_id, "training", f"Tuning failed for {name}, falling back to default training.")
                            from sklearn.base import clone
                            estimator = clone(estimator)
                            estimator.fit(Xt_train, y_train)
                else:
                    estimator.fit(Xt_train, y_train)
                train_time = time.time() - t_start

                # Measure inference time
                t_infer_start = time.time()
                preds = estimator.predict(Xt_test)
                infer_time = time.time() - t_infer_start

                if is_clf:
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
                    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
                    metrics = {
                        "accuracy": float(acc),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_weighted": float(f1)
                    }
                    primary_metric = f1
                else:
                    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                    mae = float(mean_absolute_error(y_test, preds))
                    r2 = float(r2_score(y_test, preds))
                    metrics = {
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2
                    }
                    primary_metric = r2

                mlflow.log_metrics(metrics)
                mlflow.log_param("model_name", name)

                results.append({
                    "name": name,
                    "estimator": estimator,
                    "preprocessor": preprocessor,
                    "metrics": metrics,
                    "train_time": round(train_time, 4),
                    "inference_time": round(infer_time, 4),
                    "primary_metric": primary_metric,
                    "Xt_test": Xt_test,
                })

        update_progress(experiment_id, "evaluation", "Comparing metrics and ranking models on the leaderboard.")
        
        # Rank the results
        sorted_results = sorted(results, key=lambda x: x["primary_metric"], reverse=True)
        leaderboard = []
        for i, r in enumerate(sorted_results):
            leaderboard.append({
                "rank": i + 1,
                "model_name": r["name"],
                "metrics": r["metrics"],
                "train_time": r["train_time"],
                "inference_time": r["inference_time"]
            })

        best = sorted_results[0]

        model_id = uuid.uuid4().hex
        model_path = os.path.join(MODELS_DIR, f"model_{model_id}.joblib")
        joblib.dump({
            "preprocessor": best["preprocessor"],
            "estimator": best["estimator"],
            "task": "classification" if is_clf else "regression",
            "feature_columns": numeric_cols + categorical_cols,
            "target_column": target_column,
            "label_encoder": label_encoder,
        }, model_path)

        mlflow.log_param("best_model", best["name"])
        mlflow.log_metrics({f"best_{k}": v for k, v in best["metrics"].items()})

        # Calculate Classification Curves (Confusion Matrix, ROC, PR Curves)
        curves = None
        if is_clf:
            y_prob = None
            if hasattr(best["estimator"], "predict_proba"):
                try:
                    y_prob = best["estimator"].predict_proba(best["Xt_test"])
                except Exception:
                    pass
            curves = calculate_classification_plots(
                y_test.values,
                best["estimator"].predict(best["Xt_test"]),
                y_prob,
                is_binary=(y.nunique() <= 2)
            )

        # Generate Business Insights
        insights = []
        # Find feature importance or coefficients
        importances = None
        if hasattr(best["estimator"], "feature_importances_"):
            importances = best["estimator"].feature_importances_
        elif hasattr(best["estimator"], "coef_"):
            coef = best["estimator"].coef_
            if len(coef.shape) == 1:
                importances = np.abs(coef)
            else:
                importances = np.abs(coef[0])

        top_feats = []
        if importances is not None and len(importances) == len(feature_names):
            idx_sorted = np.argsort(importances)[::-1]
            top_feats = [feature_names[i] for i in idx_sorted[:3] if i < len(feature_names)]

        if is_clf:
            cls_counts = df[target_column].value_counts(normalize=True)
            maj_class = cls_counts.index[0]
            maj_pct = round(cls_counts.iloc[0] * 100, 1)
            insights.append(f"Class Distribution: The majority class is '{maj_class}' ({maj_pct}% of cases).")
            if len(top_feats) > 0:
                insights.append(f"Predictive Factors: Features '{', '.join(top_feats)}' show the highest impact on classifying target categories.")
            insights.append(f"Model Optimization: The champion model is '{best['name']}' with an F1 score of {round(best['metrics']['f1_weighted'], 4)}.")
        else:
            if len(top_feats) > 0:
                insights.append(f"Key Drivers: Feature '{top_feats[0]}' has the highest coefficient weight, driving values of continuous target '{target_column}'.")
            insights.append(f"Fit Accuracy: The model explains {round(best['metrics']['r2'] * 100, 2)}% of the target variation on validation folds.")

        update_progress(experiment_id, "shap", "Generating SHAP tree importance explanations.")
        shap_path = None
        if best["name"] in ("random_forest", "xgboost", "lightgbm", "catboost"):
            try:
                sample = best["Xt_test"][: min(200, len(best["Xt_test"]))]
                shap_path = generate_shap_summary(best["estimator"], sample, feature_names=feature_names, model_kind="tree")
            except Exception as e:
                print(f"Failed to generate SHAP: {e}")

        update_progress(experiment_id, "completed", "Automated pipeline complete. Model registered in system registry.")

        return {
            "mlflow_run_id": parent_run.info.run_id,
            "best_model_name": best["name"],
            "metrics": {r["name"]: r["metrics"] for r in results},
            "best_metrics": best["metrics"],
            "leaderboard": leaderboard,
            "curves": curves,
            "business_insights": insights,
            "model_path": model_path,
            "shap_plot_path": shap_path,
            "task": "classification" if is_clf else "regression",
        }
