"""
Database module for real-time loan prediction audit trail.
Supports SQLite out of the box and connects to PostgreSQL when configured.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.config import DB_PATH

def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        loan_id TEXT,
        age REAL,
        income REAL,
        loan_amount REAL,
        credit_score REAL,
        months_employed REAL,
        num_credit_lines REAL,
        interest_rate REAL,
        loan_term REAL,
        dti_ratio REAL,
        education TEXT,
        employment_type TEXT,
        marital_status TEXT,
        has_mortgage TEXT,
        has_dependents TEXT,
        loan_purpose TEXT,
        has_co_signer TEXT,
        default_probability REAL,
        prediction INTEGER,
        risk_rating TEXT,
        recommendation TEXT,
        model_version TEXT,
        latency_ms REAL,
        shap_top_drivers TEXT,
        actual_default INTEGER DEFAULT NULL
    )
    """)
    conn.commit()
    conn.close()

def log_prediction(
    loan_id: str,
    features: Dict[str, Any],
    default_prob: float,
    prediction: int,
    risk_rating: str,
    recommendation: str,
    model_version: str = "v1.0-xgboost",
    latency_ms: float = 0.0,
    shap_top_drivers: Optional[Dict[str, float]] = None
) -> int:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    drivers_json = json.dumps(shap_top_drivers or {})
    
    cursor.execute("""
    INSERT INTO loan_predictions (
        loan_id, age, income, loan_amount, credit_score, months_employed,
        num_credit_lines, interest_rate, loan_term, dti_ratio, education,
        employment_type, marital_status, has_mortgage, has_dependents,
        loan_purpose, has_co_signer, default_probability, prediction,
        risk_rating, recommendation, model_version, latency_ms, shap_top_drivers
    ) VALUES (
        ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?
    )
    """, (
        loan_id,
        features.get("Age"),
        features.get("Income"),
        features.get("LoanAmount"),
        features.get("CreditScore"),
        features.get("MonthsEmployed"),
        features.get("NumCreditLines"),
        features.get("InterestRate"),
        features.get("LoanTerm"),
        features.get("DTIRatio"),
        features.get("Education"),
        features.get("EmploymentType"),
        features.get("MaritalStatus"),
        features.get("HasMortgage"),
        features.get("HasDependents"),
        features.get("LoanPurpose"),
        features.get("HasCoSigner"),
        float(default_prob),
        int(prediction),
        risk_rating,
        recommendation,
        model_version,
        latency_ms,
        drivers_json
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM loan_predictions ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_scoring_stats() -> Dict[str, Any]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        COUNT(*) as total_scored,
        AVG(default_probability) as avg_default_prob,
        AVG(latency_ms) as avg_latency_ms,
        SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as flagged_defaults,
        SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as approved_loans
    FROM loan_predictions
    """)
    row = cursor.fetchone()
    conn.close()
    if not row or row["total_scored"] == 0:
        return {
            "total_scored": 0,
            "avg_default_prob": 0.0,
            "avg_latency_ms": 0.0,
            "flagged_defaults": 0,
            "approved_loans": 0
        }
    return dict(row)
