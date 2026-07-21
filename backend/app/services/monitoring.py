import pandas as pd
import numpy as np
from scipy import stats

def calculate_drift(ref_df: pd.DataFrame, target_df: pd.DataFrame, exclude_cols: list = None) -> dict:
    """
    Computes data drift and feature-level drift between reference and target datasets.
    Uses Kolmogorov-Smirnov test for numerical features and Chi-Square test for categorical features.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify common columns (excluding those in exclude_cols)
    common_cols = [c for c in ref_df.columns if c in target_df.columns and c not in exclude_cols]
    
    drift_details = {}
    drift_count = 0
    
    for col in common_cols:
        # Check data type
        is_num_ref = pd.api.types.is_numeric_dtype(ref_df[col])
        is_num_tgt = pd.api.types.is_numeric_dtype(target_df[col])
        
        # Clean null values for testing
        ref_series = ref_df[col].dropna()
        tgt_series = target_df[col].dropna()
        
        if len(ref_series) == 0 or len(tgt_series) == 0:
            continue
            
        p_val = 1.0
        drift_detected = False
        method = "None"
        
        if is_num_ref and is_num_tgt:
            method = "Kolmogorov-Smirnov Test"
            try:
                # KS test
                res = stats.ks_2samp(ref_series, tgt_series)
                p_val = float(res.pvalue)
                drift_detected = p_val < 0.05
            except Exception:
                drift_detected = False
        else:
            method = "Chi-Square Contingency Test"
            try:
                # Align categories and construct contingency table
                ref_counts = ref_series.value_counts()
                tgt_counts = tgt_series.value_counts()
                
                # Combine unique values
                all_cats = list(set(ref_counts.index).union(set(tgt_counts.index)))
                
                # Get counts for all categories
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
                tgt_aligned = [tgt_counts.get(cat, 0) for cat in all_cats]
                
                # Chi-Square contingency
                table = np.array([ref_aligned, tgt_aligned])
                if table.shape[1] > 1 and np.sum(table) > 0:
                    res = stats.chi2_contingency(table)
                    p_val = float(res.pvalue)
                    drift_detected = p_val < 0.05
                else:
                    # Fallback to ratio difference comparison
                    ref_ratios = ref_series.value_counts(normalize=True)
                    tgt_ratios = tgt_series.value_counts(normalize=True)
                    max_diff = 0.0
                    for cat in all_cats:
                        diff = abs(ref_ratios.get(cat, 0.0) - tgt_ratios.get(cat, 0.0))
                        max_diff = max(max_diff, diff)
                    drift_detected = max_diff > 0.15 # if ratio difference > 15%, count as drift
                    p_val = 0.0 if drift_detected else 1.0
                    method = "Category Ratio Analysis"
            except Exception:
                drift_detected = False
                
        if drift_detected:
            drift_count += 1
            
        drift_details[col] = {
            "p_value": p_val,
            "drift_detected": bool(drift_detected),
            "test_method": method
        }
        
    overall_drift = False
    if len(common_cols) > 0:
        drift_ratio = drift_count / len(common_cols)
        # If more than 20% of common features show drift, we flag overall data drift
        overall_drift = drift_ratio > 0.20
        
    return {
        "drift_detected": overall_drift,
        "drifted_features_count": drift_count,
        "total_features_count": len(common_cols),
        "features": drift_details,
        "summary": f"Drift detected in {drift_count} out of {len(common_cols)} common features."
    }
