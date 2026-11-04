"""
SHAP Explainability Module for Bank Muscat Credit Underwriting.
Provides global model transparency and local real-time applicant-level
decision explanations (Adverse Action Reasons for Basel/regulatory compliance).
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, ARTIFACTS_DIR, SHAP_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CreditRiskExplainer:
    """Enterprise SHAP Explainer for Loan Default Prediction."""
    
    def __init__(self, model=None, preprocessor=None):
        if model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
        else:
            self.model = model
            
        if preprocessor is None:
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        else:
            self.preprocessor = preprocessor
            
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.explainer = shap.TreeExplainer(self.model)
        
    def compute_and_cache_global_shap(self, sample_df: pd.DataFrame, n_samples: int = 500) -> Dict[str, Any]:
        """Compute global SHAP values on a representative sample and cache results."""
        logger.info("Computing global SHAP values on %d samples...", min(n_samples, len(sample_df)))
        subset = sample_df.sample(n=min(n_samples, len(sample_df)), random_state=42)
        X_proc = self.preprocessor.transform(subset)
        
        shap_values = self.explainer.shap_values(X_proc)
        # In binary classification, if shape is 2D (samples, features)
        if isinstance(shap_values, list):
            vals = shap_values[1] # positive class
        else:
            vals = shap_values
            
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        importance_ranking = []
        for name, val in zip(self.feature_names, mean_abs_shap):
            importance_ranking.append({
                "feature": name,
                "importance": round(float(val), 4)
            })
            
        importance_ranking.sort(key=lambda x: x["importance"], reverse=True)
        
        cached_payload = {
            "feature_importance": importance_ranking,
            "base_value": round(float(self.explainer.expected_value), 4),
            "sample_shap_matrix": vals[:100].tolist(), # save first 100 for beeswarm visualization
            "sample_feature_matrix": X_proc[:100].tolist(),
            "feature_names": self.feature_names
        }
        
        joblib.dump(cached_payload, SHAP_DATA_PATH)
        logger.info("Global SHAP summary saved to %s", SHAP_DATA_PATH)
        return cached_payload

    def explain_instance(self, single_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate local SHAP explanations for an individual loan applicant."""
        X_proc = self.preprocessor.transform(single_df)
        shap_vals = self.explainer.shap_values(X_proc)
        
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals[0]
            
        base_val = float(self.explainer.expected_value)
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])
            
        feature_contributions = []
        for name, val in zip(self.feature_names, vals):
            feature_contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "impact": "Increases Risk (Default)" if val > 0 else "Reduces Risk (Safe)"
            })
            
        # Top Risk Catalysts (+ve SHAP)
        risk_drivers = sorted([f for f in feature_contributions if f["shap_value"] > 0], 
                              key=lambda x: x["shap_value"], reverse=True)[:5]
                              
        # Top Safety Anchors (-ve SHAP)
        safe_drivers = sorted([f for f in feature_contributions if f["shap_value"] < 0], 
                              key=lambda x: x["shap_value"])[:5]
                              
        # Generate underwriting rationales
        narratives = []
        raw_row = single_df.iloc[0]
        if raw_row.get("InterestRate", 0) > 15.0:
            narratives.append(f"High borrowing interest rate of {raw_row['InterestRate']:.1f}% increases debt service burden.")
        if raw_row.get("DTIRatio", 0) > 0.45:
            narratives.append(f"Debt-to-Income ratio of {raw_row['DTIRatio']:.2f} exceeds preferred underwriting limit (0.40).")
        if raw_row.get("MonthsEmployed", 0) < 12:
            narratives.append(f"Limited current employment tenure ({int(raw_row['MonthsEmployed'])} months).")
        if raw_row.get("CreditScore", 0) < 600:
            narratives.append(f"Bureau credit score ({int(raw_row['CreditScore'])}) falls into high-risk tier.")
        if raw_row.get("HasCoSigner") == "No":
            narratives.append("Lack of a guarantor or co-signer elevates unsecured loss exposure.")
            
        if not narratives:
            narratives.append("Favorable financial profile with solid income-to-debt ratio and consistent credit history.")

        return {
            "base_value": round(base_val, 4),
            "top_risk_drivers": risk_drivers,
            "top_safe_drivers": safe_drivers,
            "all_contributions": feature_contributions,
            "regulatory_adverse_actions": narratives
        }

def get_or_create_explainer() -> CreditRiskExplainer:
    return CreditRiskExplainer()
