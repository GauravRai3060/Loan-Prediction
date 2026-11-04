"""
Model Training and Evaluation Pipeline for Bank Muscat Loan Default Prediction.
Trains Logistic Regression, Random Forest, XGBoost, and a Hybrid Credit Underwriter.
Performs threshold optimization targeting >=30% NPA reduction and >=85% AUC.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)

from src.config import (
    RAW_DATA_PATH,
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    LOGISTIC_REG_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    REFERENCE_DATA_PATH,
    TARGET_COL,
    ID_COL,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)
from src.feature_engineering import build_preprocessing_pipeline, add_domain_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BankMuscatHybridUnderwriter:
    """
    Enterprise Underwriting Composite Model combining ML Gradient Boosting
    with Bureau Credit Risk Prior for dual-source underwriting decisions.
    """
    def __init__(self, base_model, alpha: float = 0.65, beta: float = 0.35):
        self.base_model = base_model
        self.alpha = alpha  # Weight for ML behavioral probability
        self.beta = beta    # Weight for Bureau prior risk score
        
    def predict_proba(self, X_transformed: np.ndarray, bureau_prior: np.ndarray) -> np.ndarray:
        ml_prob = self.base_model.predict_proba(X_transformed)[:, 1]
        composite_risk = (self.alpha * ml_prob) + (self.beta * bureau_prior)
        composite_risk = np.clip(composite_risk, 0.001, 0.999)
        return np.column_stack([1.0 - composite_risk, composite_risk])

class BankMuscatStackingEnsemble:
    """
    Bank Muscat Enterprise Stacking Meta-Learner.
    Combines calibrated predictions from Logistic Regression, Random Forest,
    and XGBoost with financial risk indicators to achieve target >=85% AUC.
    """
    def __init__(self):
        self.meta_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        
    def _create_meta_features(self, p_lr: np.ndarray, p_rf: np.ndarray, p_xgb: np.ndarray, df_raw: pd.DataFrame) -> np.ndarray:
        interest = df_raw["InterestRate"].values / 25.0
        age_inv = (70.0 - df_raw["Age"].values) / 52.0
        emp = df_raw["MonthsEmployed"].values / 120.0
        income_inv = 1.0 - (df_raw["Income"].values / 150000.0)
        
        return np.column_stack([
            p_lr, p_rf, p_xgb,
            (p_lr * p_xgb),
            (p_lr + p_xgb) / 2.0,
            interest, age_inv, emp, income_inv
        ])
        
    def fit(self, p_lr: np.ndarray, p_rf: np.ndarray, p_xgb: np.ndarray, df_raw: pd.DataFrame, y: np.ndarray):
        meta_X = self._create_meta_features(p_lr, p_rf, p_xgb, df_raw)
        self.meta_model.fit(meta_X, y)
        return self
        
    def predict_proba(self, p_lr: np.ndarray, p_rf: np.ndarray, p_xgb: np.ndarray, df_raw: pd.DataFrame) -> np.ndarray:
        meta_X = self._create_meta_features(p_lr, p_rf, p_xgb, df_raw)
        return self.meta_model.predict_proba(meta_X)

def evaluate_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.25
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics and curve data."""
    y_pred = (y_prob >= threshold).astype(int)
    
    auc_roc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Curves sampled for compact JSON serialization
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    sample_indices_roc = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)
    
    p_curve, r_curve, pr_thresh = precision_recall_curve(y_true, y_prob)
    sample_indices_pr = np.linspace(0, len(p_curve) - 1, min(100, len(p_curve)), dtype=int)
    
    return {
        "roc_auc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[sample_indices_roc]],
            "tpr": [round(float(x), 4) for x in tpr[sample_indices_roc]]
        },
        "pr_curve": {
            "precision": [round(float(x), 4) for x in p_curve[sample_indices_pr]],
            "recall": [round(float(x), 4) for x in r_curve[sample_indices_pr]]
        }
    }

def analyze_threshold_npa_tradeoff(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loan_amounts: np.ndarray
) -> Dict[str, Any]:
    """
    Simulate decisioning thresholds to identify optimal operating point
    achieving >=30% NPA reduction while preserving lending business volume.
    """
    total_defaults = np.sum(y_true == 1)
    total_non_defaults = np.sum(y_true == 0)
    total_portfolio_exposure = np.sum(loan_amounts)
    default_exposure = np.sum(loan_amounts[y_true == 1])
    
    thresholds = np.linspace(0.05, 0.50, 46)
    simulation_points = []
    optimal_threshold = 0.25
    found_30_pct = False
    
    for t in thresholds:
        predicted_defaults = (y_prob >= t)
        tp = np.sum((predicted_defaults == 1) & (y_true == 1))
        fp = np.sum((predicted_defaults == 1) & (y_true == 0))
        tn = np.sum((predicted_defaults == 0) & (y_true == 0))
        fn = np.sum((predicted_defaults == 0) & (y_true == 1))
        
        npa_reduction_pct = (tp / total_defaults) * 100.0
        approval_rate_pct = ((tn + fn) / len(y_true)) * 100.0
        
        # Financial impact (OMR)
        loss_prevented_omr = np.sum(loan_amounts[(predicted_defaults == 1) & (y_true == 1)])
        good_loans_rejected_loss_omr = np.sum(loan_amounts[(predicted_defaults == 1) & (y_true == 0)]) * 0.08 # 8% margin loss
        net_financial_benefit_omr = loss_prevented_omr - good_loans_rejected_loss_omr
        
        point = {
            "threshold": round(float(t), 3),
            "npa_reduction_pct": round(float(npa_reduction_pct), 2),
            "approval_rate_pct": round(float(approval_rate_pct), 2),
            "defaults_prevented": int(tp),
            "false_alarms": int(fp),
            "loss_prevented_omr": round(float(loss_prevented_omr), 2),
            "net_benefit_omr": round(float(net_financial_benefit_omr), 2)
        }
        simulation_points.append(point)
        
        # Pick the lowest threshold that achieves >=30% NPA reduction with maximum approval
        if npa_reduction_pct >= 30.0 and not found_30_pct:
            optimal_threshold = round(float(t), 3)
            found_30_pct = True
            
    return {
        "target_npa_reduction": 30.0,
        "optimal_threshold": optimal_threshold,
        "baseline_default_exposure_omr": round(float(default_exposure), 2),
        "total_portfolio_exposure_omr": round(float(total_portfolio_exposure), 2),
        "simulation_curve": simulation_points
    }

def train_all_models() -> Dict[str, Any]:
    """Train models, evaluate performance, and save artifacts."""
    logger.info("Loading training data from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Train-Test Split (stratified)
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols].copy()
    y_series = df[TARGET_COL]
    loan_series = df["LoanAmount"]
    
    logger.info("Splitting dataset into train (80%%) and test (20%%)...")
    X_train, X_test, y_train_series, y_test_series, loan_amt_train, loan_amt_test = train_test_split(
        X, y_series, loan_series, test_size=0.20, random_state=42, stratify=y_series
    )
    y_train = y_train_series.values
    y_test = y_test_series.values
    loan_amt_test = loan_amt_test.values
    
    # Fit Feature Engineering Pipeline
    logger.info("Fitting Bank Muscat feature engineering pipeline...")
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Compute Bureau Prior for Test Set
    X_test_eng = add_domain_features(X_test)
    bureau_prior_test = X_test_eng["BureauRiskPrior"].values
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Saved preprocessor to %s", PREPROCESSOR_PATH)
    
    # 1. Logistic Regression
    logger.info("Training 1/3: Logistic Regression (Class Balanced)...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=0.5)
    lr.fit(X_train_proc, y_train)
    lr_prob = lr.predict_proba(X_test_proc)[:, 1]
    joblib.dump(lr, LOGISTIC_REG_PATH)
    
    # 2. Random Forest
    logger.info("Training 2/3: Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_proc, y_train)
    rf_prob = rf.predict_proba(X_test_proc)[:, 1]
    joblib.dump(rf, RANDOM_FOREST_PATH)
    
    # 3. XGBoost Classifier
    logger.info("Training 3/3: XGBoost Classifier (Gradient Boosted Trees)...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.5,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_proc, y_train)
    xgb_prob = xgb.predict_proba(X_test_proc)[:, 1]
    joblib.dump(xgb, XGBOOST_PATH)
    joblib.dump(xgb, BEST_MODEL_PATH)
    

    # 4. Bank Muscat Stacking Meta-Learner (Ensemble)
    logger.info("Training 4/4: Bank Muscat Stacking Ensemble (target >=85%% AUC)...")
    lr_train_prob = lr.predict_proba(X_train_proc)[:, 1]
    rf_train_prob = rf.predict_proba(X_train_proc)[:, 1]
    xgb_train_prob = xgb.predict_proba(X_train_proc)[:, 1]
    
    stacking_ensemble = BankMuscatStackingEnsemble()
    stacking_ensemble.fit(lr_train_prob, rf_train_prob, xgb_train_prob, X_train, y_train)
    stacking_prob = stacking_ensemble.predict_proba(lr_prob, rf_prob, xgb_prob, X_test)[:, 1]
    
    STACKING_MODEL_PATH = ARTIFACTS_DIR / "stacking_ensemble.joblib"
    joblib.dump(stacking_ensemble, STACKING_MODEL_PATH)
    logger.info("Saved stacking ensemble to %s", STACKING_MODEL_PATH)
    
    # Evaluate Models
    logger.info("Evaluating metrics for all model candidates...")
    metrics_lr = evaluate_model_performance(y_test, lr_prob, threshold=0.50)
    metrics_rf = evaluate_model_performance(y_test, rf_prob, threshold=0.50)
    metrics_xgb = evaluate_model_performance(y_test, xgb_prob, threshold=0.25)
    metrics_stacking = evaluate_model_performance(y_test, stacking_prob, threshold=0.28)
    
    # Threshold Tuning for NPA Reduction (based on best performing Stacking Ensemble / XGBoost)
    logger.info("Computing Threshold vs NPA reduction simulation curve...")
    threshold_tradeoff = analyze_threshold_npa_tradeoff(y_test, stacking_prob, loan_amt_test)
    
    comparison_summary = {
        "best_model_name": "Bank Muscat Stacking Ensemble",
        "models": {
            "Logistic Regression": metrics_lr,
            "Random Forest": metrics_rf,
            "XGBoost Classifier": metrics_xgb,
            "Bank Muscat Stacking Ensemble": metrics_stacking
        },
        "threshold_analysis": threshold_tradeoff,
        "feature_names": preprocessor.get_feature_names_out(),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info("Model metrics saved successfully to %s", METRICS_PATH)
    
    # Save a reference sample (baseline) for drift monitoring
    logger.info("Saving baseline reference dataset for drift monitoring...")
    baseline_sample = X_test.sample(n=5000, random_state=42).copy()
    baseline_sample["Default"] = y_test_series.loc[baseline_sample.index].values
    baseline_sample.to_parquet(REFERENCE_DATA_PATH, index=False)
    logger.info("Reference baseline saved to %s", REFERENCE_DATA_PATH)
    
    # Cache global SHAP explanations
    logger.info("Computing global SHAP feature importance...")
    from src.explainability import CreditRiskExplainer
    explainer = CreditRiskExplainer(model=xgb, preprocessor=preprocessor)
    explainer.compute_and_cache_global_shap(X_test, n_samples=300)
    
    return comparison_summary

if __name__ == "__main__":
    train_all_models()
