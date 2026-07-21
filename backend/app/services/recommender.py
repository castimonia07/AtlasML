import pandas as pd


def parse_problem_statement(text: str | None) -> tuple[str | None, str | None, float | None]:
    """Parse key concepts from the project description / problem statement."""
    if not text:
        return None, None, None
    text_lower = text.lower()

    # Time Series keywords
    ts_keywords = [
        "forecast", "time series", "predict sales", "predict traffic", "over time",
        "weekly", "monthly", "daily", "temporal", "stock price", "forecasting"
    ]
    for kw in ts_keywords:
        if kw in text_lower:
            return "time_series", f"Problem statement mentions '{kw}', indicating a time-series forecasting task.", 0.95

    # Unsupervised keywords
    unsup_keywords = [
        "cluster", "group", "segment", "pattern", "structure", "unsupervised",
        "clustering", "dimensionality reduction", "pca", "anomaly detection"
    ]
    for kw in unsup_keywords:
        if kw in text_lower:
            return "unsupervised", f"Problem statement mentions '{kw}', recommending unsupervised clustering to discover patterns.", 0.88

    # Supervised classification keywords
    class_keywords = [
        "classify", "classification", "churn", "spam", "sentiment", "category",
        "class", "yes/no", "positive/negative", "detect fraud"
    ]
    for kw in class_keywords:
        if kw in text_lower:
            return "supervised_clf", f"Problem statement mentions '{kw}', suggesting a supervised classification task.", 0.92

    # Supervised regression keywords
    reg_keywords = [
        "predict price", "predict value", "regression", "score", "amount",
        "salary", "house price", "continuous"
    ]
    for kw in reg_keywords:
        if kw in text_lower:
            return "supervised_reg", f"Problem statement mentions '{kw}', suggesting a supervised regression task.", 0.92

    # Generic predict/supervised keywords
    if "predict" in text_lower or "supervised" in text_lower:
        return "supervised", "Problem statement indicates a prediction task, recommending a supervised learning pipeline.", 0.80

    return None, None, None


def recommend_workflow(
    df: pd.DataFrame,
    target_column: str | None,
    date_column: str | None,
    problem_statement: str | None = None
) -> dict:
    """Recommendation engine returning workflow_type, confidence, reason, candidate_models and suggested_metric."""

    # 1. Check the Problem Statement Parser first
    workflow, reason, confidence = parse_problem_statement(problem_statement)
    
    # Map raw workflow suggestion to standard database workflow_types
    mapped_workflow = workflow
    if workflow == "supervised_clf" or workflow == "supervised_reg":
        mapped_workflow = "supervised"

    if mapped_workflow:
        # Validate that the workflow is feasible
        if mapped_workflow == "time_series" and not date_column:
            return {
                "workflow_type": "supervised",
                "reason": "Problem statement suggested 'time_series', but no date column was found. Falling back to supervised learning.",
                "confidence": 0.70,
                "candidate_models": ["Logistic Regression", "Random Forest Classifier", "XGBoost", "LightGBM", "CatBoost"],
                "suggested_metric": "F1-Score"
            }
            
        # Refine models & metrics based on statement parser output
        if workflow == "supervised_clf":
            # Check target imbalance if target exists in dataset
            metric = "F1-Score"
            reason_suffix = " (Optimized for classification tasks)"
            if target_column and target_column in df.columns:
                counts = df[target_column].value_counts(normalize=True)
                if len(counts) > 1 and counts.iloc[0] > 0.65:
                    metric = "Weighted F1-Score"
                    reason_suffix = f" (F1-Score chosen due to class imbalance: {round(counts.iloc[0]*100, 1)}% majority class)"
            return {
                "workflow_type": "supervised",
                "reason": reason + reason_suffix,
                "confidence": confidence,
                "candidate_models": ["Logistic Regression", "Random Forest Classifier", "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier"],
                "suggested_metric": metric
            }
        elif workflow == "supervised_reg":
            return {
                "workflow_type": "supervised",
                "reason": reason + " (Optimized for regression/continuous tasks)",
                "confidence": confidence,
                "candidate_models": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor"],
                "suggested_metric": "R2-Score"
            }
        elif workflow == "unsupervised":
            return {
                "workflow_type": "unsupervised",
                "reason": reason,
                "confidence": confidence,
                "candidate_models": ["K-Means Clustering", "PCA Projection"],
                "suggested_metric": "Silhouette Score"
            }
        elif workflow == "time_series":
            return {
                "workflow_type": "time_series",
                "reason": reason,
                "confidence": confidence,
                "candidate_models": ["Random Forest Lag Forecaster"],
                "suggested_metric": "RMSE (Root Mean Squared Error)"
            }

    # 2. Fallback to Dataset Heuristics if no problem statement keywords match
    if not target_column:
        return {
            "workflow_type": "unsupervised",
            "reason": "No target column was selected, so clustering and dimensionality reduction are recommended to explore structure in the data.",
            "confidence": 0.75,
            "candidate_models": ["K-Means Clustering", "PCA Projection"],
            "suggested_metric": "Silhouette Score"
        }

    # Time series check: only recommend if the date column name strongly implies a time index
    if date_column and target_column:
        date_lower = date_column.lower()
        is_strong_time_index = any(
            kw in date_lower for kw in ["date", "timestamp", "time", "ds", "epoch", "datetime"]
        )
        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        if is_strong_time_index and is_numeric:
            return {
                "workflow_type": "time_series",
                "reason": (
                    f"A strong time-index column ('{date_column}') and numeric target "
                    f"('{target_column}') suggest a time-series forecasting problem."
                ),
                "confidence": 0.80,
                "candidate_models": ["Random Forest Lag Forecaster"],
                "suggested_metric": "RMSE (Root Mean Squared Error)"
            }

    # Standard supervised check
    nunique = df[target_column].nunique()
    is_numeric = pd.api.types.is_numeric_dtype(df[target_column])

    if is_numeric and nunique > 20:
        return {
            "workflow_type": "supervised",
            "reason": (
                f"Target '{target_column}' is numeric with {nunique} distinct values, "
                "indicating a regression problem."
            ),
            "confidence": 0.90,
            "candidate_models": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor"],
            "suggested_metric": "R2-Score"
        }

    # Imbalance check for classification default
    metric = "F1-Score"
    counts = df[target_column].value_counts(normalize=True)
    if len(counts) > 1 and counts.iloc[0] > 0.65:
        metric = "Weighted F1-Score"
        reason_text = (
            f"Target '{target_column}' has {nunique} distinct values, indicating a "
            f"classification problem. Metric F1-Score is chosen due to class imbalance ({round(counts.iloc[0]*100, 1)}% majority)."
        )
    else:
        reason_text = (
            f"Target '{target_column}' has {nunique} distinct values, indicating a "
            "classification problem."
        )

    return {
        "workflow_type": "supervised",
        "reason": reason_text,
        "confidence": 0.88,
        "candidate_models": ["Logistic Regression", "Random Forest Classifier", "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier"],
        "suggested_metric": metric
    }
