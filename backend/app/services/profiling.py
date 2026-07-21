import pandas as pd
import numpy as np


def calculate_health_report(df: pd.DataFrame, target_column: str | None) -> dict:
    """Calculates dataset health statistics, warnings, recommendations, and quality score."""
    warnings = []
    recommendations = []
    quality_score = 100

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # 1. Missing Values check
    null_counts = df.isna().sum()
    total_nulls = int(null_counts.sum())
    missing_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
    
    if missing_pct > 0:
        # Deduct up to 25 points
        deduction = min(25, int(missing_pct * 0.8))
        quality_score -= deduction
        warnings.append(f"{round(missing_pct, 1)}% of cells in the dataset are missing.")
        recommendations.append("Apply imputation (e.g., median for numeric, mode for categorical) to fill missing values.")

    for col in df.columns:
        col_null_pct = (df[col].isna().sum() / n_rows) * 100 if n_rows > 0 else 0
        if col_null_pct > 25:
            warnings.append(f"Column '{col}' has a high missing rate of {round(col_null_pct, 1)}%.")
            if col_null_pct > 80:
                recommendations.append(f"Consider dropping column '{col}' since it has over 80% missing data.")

    # 2. Duplicate rows check
    n_duplicates = int(df.duplicated().sum())
    duplicate_pct = (n_duplicates / n_rows) * 100 if n_rows > 0 else 0
    if n_duplicates > 0:
        # Deduct up to 15 points
        deduction = min(15, int(duplicate_pct * 1.5) + 1)
        quality_score -= deduction
        warnings.append(f"Detected {n_duplicates} duplicate rows ({round(duplicate_pct, 1)}% of dataset).")
        recommendations.append("Drop duplicate rows during preprocessing to prevent model training bias.")

    # 3. Constant/Zero Variance columns check
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        deduction = min(15, len(constant_cols) * 3)
        quality_score -= deduction
        warnings.append(f"Columns {constant_cols} are constant (contain only a single unique value).")
        recommendations.append(f"Drop constant columns {constant_cols} as they carry zero variance and offer no predictive power.")

    # 4. Class Imbalance check in target column
    is_imbalanced = False
    imbalance_details = {}
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        nunique_target = target_series.nunique()
        # Only check imbalance for classification (low cardinality targets)
        if nunique_target > 1 and nunique_target <= 10:
            counts = target_series.value_counts(normalize=True)
            majority_class = counts.index[0]
            majority_ratio = counts.iloc[0]
            if majority_ratio > 0.70:
                is_imbalanced = True
                # Deduct up to 15 points
                deduction = min(15, int((majority_ratio - 0.7) * 50))
                quality_score -= deduction
                warnings.append(
                    f"High class imbalance in target '{target_column}': "
                    f"majority class '{majority_class}' represents {round(majority_ratio * 100, 1)}% of samples."
                )
                recommendations.append(
                    f"Enable class weighting or utilize sampling techniques (like SMOTE) for training "
                    f"due to imbalanced target '{target_column}'."
                )
                imbalance_details = {
                    "majority_class": str(majority_class),
                    "majority_ratio": float(majority_ratio),
                }

    # Clamp quality score
    quality_score = max(0, min(100, quality_score))

    return {
        "quality_score": quality_score,
        "missing_percentage": round(missing_pct, 2),
        "duplicate_percentage": round(duplicate_pct, 2),
        "is_imbalanced": is_imbalanced,
        "imbalance_details": imbalance_details,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def profile_dataframe(df: pd.DataFrame, target_column: str | None = None) -> dict:
    n_rows, n_cols = df.shape

    dtypes = {c: str(df[c].dtype) for c in df.columns}
    null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        numeric_summary = {
            col: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in stats.items()}
            for col, stats in desc.items()
        }

    date_columns = detect_date_columns(df)
    
    # Auto Target Suggestion
    suggested_target = target_column or suggest_target(df, date_columns)
    
    # Calculate dataset health check report
    health_report = calculate_health_report(df, suggested_target)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "null_counts": null_counts,
        "duplicate_rows": duplicate_rows,
        "numeric_summary": numeric_summary,
        "detected_date_columns": date_columns,
        "suggested_target": suggested_target,
        "health_report": health_report,
    }


def detect_date_columns(df: pd.DataFrame) -> list:
    date_cols = []
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            date_cols.append(c)
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().head(20)
            try:
                pd.to_datetime(sample, errors="raise", format="mixed")
                date_cols.append(c)
            except Exception:
                pass
    return date_cols


def suggest_target(df: pd.DataFrame, date_columns: list) -> str | None:
    """Heuristic: prefer churn, target, label, target-like columns, or default to the last column."""
    
    # Check for exact matches first
    lower_cols = [c.lower() for c in df.columns]
    for target_word in ["churn", "target", "label", "output", "class", "price", "y"]:
        if target_word in lower_cols:
            idx = lower_cols.index(target_word)
            return df.columns[idx]
            
    candidates = [c for c in df.columns if c not in date_columns]
    candidates = [c for c in candidates if "id" != c.lower() and not c.lower().endswith("_id")]
    if not candidates:
        return None

    # Prefer the last column if it looks like a reasonable target
    last = candidates[-1]
    if df[last].nunique() < max(20, len(df) * 0.5):
        return last
    return candidates[-1]
