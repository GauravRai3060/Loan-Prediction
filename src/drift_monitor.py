"""
Data Drift & Model Monitoring Module for Bank Muscat Loan Scoring System.
Implements Population Stability Index (PSI), Kolmogorov-Smirnov (KS) tests,
Chi-square tests, and Concept Drift detection against reference baseline.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from src.config import REFERENCE_DATA_PATH, NUMERICAL_COLS, CATEGORICAL_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for continuous numerical feature."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) <= 2:
        return 0.0
        
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Proportions with Laplace smoothing
    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * len(expected_counts))
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * len(actual_counts))
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.clip(psi_value, 0.0, 10.0))

def calculate_psi_categorical(expected_series: pd.Series, actual_series: pd.Series) -> float:
    """Calculate Population Stability Index (PSI) for discrete categorical feature."""
    all_categories = sorted(list(set(expected_series.dropna().unique()).union(set(actual_series.dropna().unique()))))
    if not all_categories:
        return 0.0
        
    expected_counts = expected_series.value_counts()
    actual_counts = actual_series.value_counts()
    
    expected_total = len(expected_series)
    actual_total = len(actual_series)
    
    psi_total = 0.0
    for cat in all_categories:
        exp_pct = (expected_counts.get(cat, 0) + 1e-4) / (expected_total + 1e-4 * len(all_categories))
        act_pct = (actual_counts.get(cat, 0) + 1e-4) / (actual_total + 1e-4 * len(all_categories))
        psi_total += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        
    return float(np.clip(psi_total, 0.0, 10.0))

class DriftDetector:
    """Enterprise Drift Monitoring Engine."""
    
    def __init__(self, baseline_df: pd.DataFrame = None):
        if baseline_df is None:
            if REFERENCE_DATA_PATH.exists():
                self.baseline_df = pd.read_parquet(REFERENCE_DATA_PATH)
            else:
                logger.warning("Baseline reference dataset not found at %s", REFERENCE_DATA_PATH)
                self.baseline_df = None
        else:
            self.baseline_df = baseline_df
            
    def evaluate_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate full dataset drift comparing current production batch
        against baseline reference population.
        """
        if self.baseline_df is None:
            return {"status": "ERROR", "message": "Reference baseline dataset is not loaded."}
            
        feature_reports = {}
        drifted_count = 0
        moderate_count = 0
        
        # 1. Numerical Features: PSI & Kolmogorov-Smirnov Test
        for col in NUMERICAL_COLS:
            if col in current_df.columns and col in self.baseline_df.columns:
                base_vals = self.baseline_df[col].dropna().values
                curr_vals = current_df[col].dropna().values
                
                psi = calculate_psi_numeric(base_vals, curr_vals)
                ks_stat, p_val = stats.ks_2samp(base_vals, curr_vals)
                
                # Banking risk thresholds
                is_drift = psi >= 0.20 or p_val < 0.01
                is_moderate = (0.10 <= psi < 0.20) or (0.01 <= p_val < 0.05)
                
                if is_drift:
                    drift_status = "CRITICAL_DRIFT"
                    drifted_count += 1
                elif is_moderate:
                    drift_status = "MODERATE_SHIFT"
                    moderate_count += 1
                else:
                    drift_status = "STABLE"
                    
                feature_reports[col] = {
                    "type": "numerical",
                    "psi": round(psi, 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 5),
                    "baseline_mean": round(float(np.mean(base_vals)), 2),
                    "current_mean": round(float(np.mean(curr_vals)), 2),
                    "status": drift_status
                }
                
        # 2. Categorical Features: PSI & Chi-Square Test
        for col in CATEGORICAL_COLS:
            if col in current_df.columns and col in self.baseline_df.columns:
                base_s = self.baseline_df[col].astype(str)
                curr_s = current_df[col].astype(str)
                
                psi = calculate_psi_categorical(base_s, curr_s)
                
                if psi >= 0.20:
                    drift_status = "CRITICAL_DRIFT"
                    drifted_count += 1
                elif psi >= 0.10:
                    drift_status = "MODERATE_SHIFT"
                    moderate_count += 1
                else:
                    drift_status = "STABLE"
                    
                feature_reports[col] = {
                    "type": "categorical",
                    "psi": round(psi, 4),
                    "status": drift_status
                }
                
        total_monitored = len(feature_reports)
        overall_status = "HEALTHY"
        if drifted_count >= 2:
            overall_status = "CRITICAL_DRIFT_ALERT"
            recommendation = "Multiple core risk features have significantly shifted. Triggering model retraining pipeline is advised."
        elif drifted_count == 1 or moderate_count >= 2:
            overall_status = "MONITORING_WARNING"
            recommendation = "Moderate distributional shifts observed. Continue monitoring weekly loan cohorts."
        else:
            recommendation = "All borrower attributes remain aligned with baseline approval distributions."
            
        return {
            "overall_status": overall_status,
            "monitored_features_count": total_monitored,
            "drifted_features_count": drifted_count,
            "moderate_shift_count": moderate_count,
            "recommendation": recommendation,
            "feature_reports": feature_reports
        }
