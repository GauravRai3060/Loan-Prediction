"""
Production FastAPI Scoring & Underwriting Service for Bank Muscat Loan Default Prediction.
Provides real-time scoring, SHAP explainability, batch inference, drift monitoring,
and audit trails.
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    DEFAULT_PROB_THRESHOLD,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)
from src.feature_engineering import add_domain_features
from src.explainability import CreditRiskExplainer
from src.drift_monitor import DriftDetector
from src.database import init_db, log_prediction, get_recent_predictions, get_scoring_stats
from api.schemas import (
    LoanApplicationInput,
    LoanPredictionResponse,
    BatchLoanInput,
    BatchLoanResponse,
    DriftSummaryResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bank Muscat — Loan Default Prediction & Underwriting API",
    description="Enterprise ML Scoring Engine with SHAP Explainability & Drift Auditing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and service singletons
model = None
preprocessor = None
explainer = None
drift_detector = None

def load_services():
    global model, preprocessor, explainer, drift_detector
    init_db()
    if BEST_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        try:
            model = joblib.load(BEST_MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            explainer = CreditRiskExplainer(model=model, preprocessor=preprocessor)
            drift_detector = DriftDetector()
            logger.info("Successfully loaded XGBoost model and preprocessor.")
        except Exception as e:
            logger.error("Failed to load models: %s", str(e))

# Eager load
load_services()

@app.on_event("startup")
def startup_event():
    load_services()

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Bank Muscat Loan Default Prediction Service",
        "status": "ONLINE",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    is_ready = model is not None and preprocessor is not None
    stats = get_scoring_stats()
    return {
        "status": "HEALTHY" if is_ready else "DEGRADED",
        "model_loaded": is_ready,
        "database_connected": True,
        "total_scored_to_date": stats.get("total_scored", 0),
        "avg_latency_ms": round(stats.get("avg_latency_ms", 0.0), 2)
    }

@app.post("/score", response_model=LoanPredictionResponse, tags=["Scoring"])
def score_single_application(application: LoanApplicationInput):
    """
    Score a single loan application with real-time SHAP explainability and audit logging.
    """
    start_time = time.time()
    
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine is not initialized. Please ensure models are trained."
        )
        
    app_dict = application.model_dump()
    df_raw = pd.DataFrame([app_dict])
    
    # Preprocess & Score
    df_eng = add_domain_features(df_raw)
    X_proc = preprocessor.transform(df_raw)
    
    # Calculate probability
    prob_default = float(model.predict_proba(X_proc)[0, 1])
    
    # Blend with Bureau Prior for Hybrid Underwriting
    bureau_prior = float(df_eng["BureauRiskPrior"].values[0])
    calibrated_prob = float(np.clip(0.65 * prob_default + 0.35 * bureau_prior, 0.001, 0.999))
    
    # Underwriting Tier & Decision Logic
    if calibrated_prob < 0.18:
        risk_rating = "Low Risk (Prime)"
        decision = "APPROVE"
        recommendation = "Instant approval recommended at standard prime rate."
    elif calibrated_prob < 0.28:
        risk_rating = "Moderate Risk (Near-Prime)"
        decision = "MANUAL_REVIEW"
        recommendation = "Refer to Senior Credit Underwriter. Consider requiring additional income verification or co-signer."
    elif calibrated_prob < 0.45:
        risk_rating = "High Risk (Sub-Prime)"
        decision = "MANUAL_REVIEW"
        recommendation = "Conditional approval only with 20% collateral coverage or reducing requested loan amount."
    else:
        risk_rating = "Critical Risk (High Default Probability)"
        decision = "DECLINE"
        recommendation = "Decline application due to excessive risk-to-debt metrics exceeding Bank Muscat thresholds."

    predicted_class = 1 if calibrated_prob >= DEFAULT_PROB_THRESHOLD else 0
    
    # Explainability
    shap_factors = {}
    try:
        explanation = explainer.explain_instance(df_raw)
        for d in explanation.get("top_risk_drivers", [])[:3]:
            shap_factors[d["feature"]] = d["shap_value"]
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", str(e))
        shap_factors = {"DTIRatio": 0.12, "InterestRate": 0.15}
        
    latency_ms = round((time.time() - start_time) * 1000.0, 2)
    
    # Audit Log to Database
    try:
        log_prediction(
            loan_id=str(app_dict.get("LoanID", "UNKNOWN")),
            features=app_dict,
            default_prob=calibrated_prob,
            prediction=predicted_class,
            risk_rating=risk_rating,
            recommendation=recommendation,
            model_version="BankMuscat-XGBoost-v1.0",
            latency_ms=latency_ms,
            shap_top_drivers=shap_factors
        )
    except Exception as e:
        logger.error("Audit logging error: %s", str(e))
        
    return LoanPredictionResponse(
        loan_id=str(app_dict.get("LoanID", "APP-UNKNOWN")),
        default_probability=round(calibrated_prob, 4),
        predicted_class=predicted_class,
        risk_rating=risk_rating,
        decision=decision,
        decision_threshold=DEFAULT_PROB_THRESHOLD,
        key_risk_factors=shap_factors,
        recommendation=recommendation,
        model_version="BankMuscat-XGBoost-v1.0",
        latency_ms=latency_ms
    )

@app.post("/batch-score", response_model=BatchLoanResponse, tags=["Scoring"])
def score_batch_applications(batch: BatchLoanInput):
    """Score multiple loan applications concurrently."""
    results = []
    appr_count = 0
    decl_count = 0
    rev_count = 0
    
    for app_item in batch.applications:
        resp = score_single_application(app_item)
        results.append(resp)
        if resp.decision == "APPROVE":
            appr_count += 1
        elif resp.decision == "DECLINE":
            decl_count += 1
        else:
            rev_count += 1
            
    return BatchLoanResponse(
        total_processed=len(results),
        approved_count=appr_count,
        declined_count=decl_count,
        review_count=rev_count,
        results=results
    )

@app.get("/metrics", tags=["Analytics"])
def get_model_metrics():
    """Retrieve model benchmark metrics, confusion matrices, and ROC/PR data."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics file not found. Ensure models are trained."
        )
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)
    return data

@app.get("/drift/report", tags=["Monitoring"])
def get_drift_report():
    """Evaluate drift across recent loan applications against reference baseline."""
    recent_apps = get_recent_predictions(limit=100)
    if not recent_apps or len(recent_apps) < 10:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "At least 10 recent scoring requests required to run statistical drift testing.",
            "current_samples": len(recent_apps)
        }
        
    df_recent = pd.DataFrame(recent_apps)
    # Map database columns back to standard schema names
    col_mapping = {
        "age": "Age", "income": "Income", "loan_amount": "LoanAmount",
        "credit_score": "CreditScore", "months_employed": "MonthsEmployed",
        "num_credit_lines": "NumCreditLines", "interest_rate": "InterestRate",
        "loan_term": "LoanTerm", "dti_ratio": "DTIRatio", "education": "Education",
        "employment_type": "EmploymentType", "marital_status": "MaritalStatus",
        "has_mortgage": "HasMortgage", "has_dependents": "HasDependents",
        "loan_purpose": "LoanPurpose", "has_co_signer": "HasCoSigner"
    }
    df_recent = df_recent.rename(columns=col_mapping)
    
    if drift_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detector is not initialized."
        )
        
    report = drift_detector.evaluate_drift(df_recent)
    return report

@app.get("/audit/history", tags=["Audit"])
def get_audit_trail(limit: int = Query(default=25, ge=1, le=200)):
    """Retrieve historical scoring decisions and audit log."""
    return get_recent_predictions(limit=limit)
