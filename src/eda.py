"""
Exploratory Data Analysis (EDA) module for Bank Muscat Loan Default Dataset.
Computes summary statistics, distributions, correlations, Information Value (IV),
and default rate breakdowns across customer demographic segments.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.config import RAW_DATA_PATH, EDA_SUMMARY_PATH, ARTIFACTS_DIR, NUMERICAL_COLS, CATEGORICAL_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> Dict[str, Any]:
    """Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature."""
    temp = df[[feature, target]].copy()
    
    if pd.api.types.is_numeric_dtype(temp[feature]) and temp[feature].nunique() > bins:
        temp['bin'] = pd.qcut(temp[feature], q=bins, duplicates='drop').astype(str)
    else:
        temp['bin'] = temp[feature].astype(str)
        
    grouped = temp.groupby('bin', observed=False).agg(
        total=(target, 'count'),
        defaults=(target, 'sum')
    ).reset_index()
    
    grouped['non_defaults'] = grouped['total'] - grouped['defaults']
    
    total_defaults = grouped['defaults'].sum()
    total_non_defaults = grouped['non_defaults'].sum()
    
    if total_defaults == 0 or total_non_defaults == 0:
        return {"iv": 0.0, "bins": []}
        
    # Smoothing to avoid div by zero or log of zero
    grouped['pct_default'] = (grouped['defaults'] + 0.5) / (total_defaults + 1.0)
    grouped['pct_non_default'] = (grouped['non_defaults'] + 0.5) / (total_non_defaults + 1.0)
    
    grouped['woe'] = np.log(grouped['pct_non_default'] / grouped['pct_default'])
    grouped['iv_contribution'] = (grouped['pct_non_default'] - grouped['pct_default']) * grouped['woe']
    
    total_iv = float(grouped['iv_contribution'].sum())
    
    bin_details = []
    for _, row in grouped.iterrows():
        bin_details.append({
            "bin": str(row['bin']),
            "total_count": int(row['total']),
            "default_count": int(row['defaults']),
            "default_rate": round(float(row['defaults'] / row['total']), 4),
            "woe": round(float(row['woe']), 4),
            "iv_contribution": round(float(row['iv_contribution']), 4)
        })
        
    return {
        "iv": round(total_iv, 4),
        "strength": (
            "Suspicious / Overfitted (>0.5)" if total_iv > 0.5 else
            "Strong Predictor (0.3 - 0.5)" if total_iv >= 0.3 else
            "Medium Predictor (0.1 - 0.3)" if total_iv >= 0.1 else
            "Weak Predictor (0.02 - 0.1)" if total_iv >= 0.02 else
            "Not Predictive (<0.02)"
        ),
        "bins": bin_details
    }

def run_eda(save_artifacts: bool = True) -> Dict[str, Any]:
    """Execute complete exploratory data analysis on the dataset."""
    logger.info("Loading raw dataset from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    
    total_records = len(df)
    target_counts = df['Default'].value_counts().to_dict()
    default_rate = float(df['Default'].mean())
    
    logger.info("Total records: %d, Default Rate: %.2f%%", total_records, default_rate * 100)
    
    # 1. Numerical Statistics
    num_summary = {}
    for col in NUMERICAL_COLS:
        num_summary[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "p25": round(float(df[col].quantile(0.25)), 2),
            "median": round(float(df[col].median()), 2),
            "p75": round(float(df[col].quantile(0.75)), 2),
            "max": round(float(df[col].max()), 2),
            "skew": round(float(df[col].skew()), 2)
        }
        
    # 2. Categorical Breakdowns & Default Rates
    cat_summary = {}
    for col in CATEGORICAL_COLS:
        cat_stats = df.groupby(col).agg(
            count=('Default', 'count'),
            defaults=('Default', 'sum'),
            default_rate=('Default', 'mean')
        ).reset_index()
        
        cat_summary[col] = [
            {
                "category": str(r[col]),
                "count": int(r['count']),
                "pct_of_total": round(float(r['count'] / total_records * 100), 2),
                "defaults": int(r['defaults']),
                "default_rate": round(float(r['default_rate']), 4)
            }
            for _, r in cat_stats.iterrows()
        ]
        
    # 3. Correlations with Target
    correlations = {}
    for col in NUMERICAL_COLS:
        correlations[col] = round(float(df[col].corr(df['Default'])), 4)
    # Sort by absolute correlation
    correlations = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))
    
    # 4. Information Value (IV) for All Features
    logger.info("Calculating Information Value (IV) & WoE for all attributes...")
    iv_analysis = {}
    for col in NUMERICAL_COLS + CATEGORICAL_COLS:
        iv_analysis[col] = calculate_woe_iv(df, col, 'Default')
        
    # Sort IV rankings
    iv_rankings = {
        k: {"iv": v["iv"], "strength": v["strength"]}
        for k, v in sorted(iv_analysis.items(), key=lambda item: item[1]["iv"], reverse=True)
    }
    
    eda_results = {
        "dataset_summary": {
            "total_records": total_records,
            "non_defaults": int(target_counts.get(0, 0)),
            "defaults": int(target_counts.get(1, 0)),
            "default_rate": round(default_rate, 4),
            "default_percentage": round(default_rate * 100, 2),
            "missing_values": int(df.isnull().sum().sum()),
            "feature_count": len(NUMERICAL_COLS) + len(CATEGORICAL_COLS)
        },
        "numerical_statistics": num_summary,
        "categorical_statistics": cat_summary,
        "correlations_with_default": correlations,
        "iv_rankings": iv_rankings,
        "detailed_woe": {col: iv_analysis[col]["bins"] for col in iv_analysis}
    }
    
    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(EDA_SUMMARY_PATH, "w") as f:
            json.dump(eda_results, f, indent=2)
        logger.info("EDA summary saved successfully to %s", EDA_SUMMARY_PATH)
        
    return eda_results

if __name__ == "__main__":
    run_eda()
