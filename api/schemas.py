"""
Pydantic schemas for Bank Muscat Loan Scoring API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class LoanApplicationInput(BaseModel):
    LoanID: Optional[str] = Field(default="APP-TEST-001", description="Unique Application or Loan ID")
    Age: int = Field(..., ge=18, le=100, description="Applicant age in years", example=35)
    Income: float = Field(..., ge=1000, description="Annual income (OMR or USD equivalent)", example=65000.0)
    LoanAmount: float = Field(..., ge=500, description="Requested loan principal amount", example=25000.0)
    CreditScore: int = Field(..., ge=300, le=850, description="Credit Bureau Score (300-850)", example=680)
    MonthsEmployed: int = Field(..., ge=0, le=600, description="Number of months in current employment", example=36)
    NumCreditLines: int = Field(..., ge=1, le=10, description="Active credit lines / cards", example=3)
    InterestRate: float = Field(..., ge=0.5, le=40.0, description="Annual Interest Rate (%)", example=9.5)
    LoanTerm: int = Field(..., ge=6, le=120, description="Loan repayment term in months", example=36)
    DTIRatio: float = Field(..., ge=0.0, le=1.5, description="Debt-to-Income ratio (0.0 - 1.0+)", example=0.35)
    Education: str = Field(..., description="High School, Bachelor's, Master's, PhD", example="Bachelor's")
    EmploymentType: str = Field(..., description="Full-time, Part-time, Self-employed, Unemployed", example="Full-time")
    MaritalStatus: str = Field(..., description="Single, Married, Divorced", example="Married")
    HasMortgage: str = Field(..., description="Yes or No", example="No")
    HasDependents: str = Field(..., description="Yes or No", example="Yes")
    LoanPurpose: str = Field(..., description="Auto, Business, Education, Home, Other", example="Home")
    HasCoSigner: str = Field(..., description="Yes or No", example="No")

    class Config:
        json_schema_extra = {
            "example": {
                "LoanID": "BM-2026-90412",
                "Age": 38,
                "Income": 75000.0,
                "LoanAmount": 30000.0,
                "CreditScore": 710,
                "MonthsEmployed": 48,
                "NumCreditLines": 2,
                "InterestRate": 8.5,
                "LoanTerm": 36,
                "DTIRatio": 0.28,
                "Education": "Bachelor's",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Married",
                "HasMortgage": "No",
                "HasDependents": "Yes",
                "LoanPurpose": "Home",
                "HasCoSigner": "No"
            }
        }

class LoanPredictionResponse(BaseModel):
    loan_id: str
    default_probability: float
    predicted_class: int
    risk_rating: str # Low Risk, Medium Risk, High Risk, Critical Risk
    decision: str    # APPROVE, MANUAL_REVIEW, DECLINE
    decision_threshold: float
    key_risk_factors: Dict[str, float]
    recommendation: str
    model_version: str
    latency_ms: float

class BatchLoanInput(BaseModel):
    applications: List[LoanApplicationInput]

class BatchLoanResponse(BaseModel):
    total_processed: int
    approved_count: int
    declined_count: int
    review_count: int
    results: List[LoanPredictionResponse]

class DriftSummaryResponse(BaseModel):
    status: str
    dataset_timestamp: str
    num_features_monitored: int
    drifted_features_count: int
    dataset_drift_detected: bool
    feature_metrics: Dict[str, Dict[str, Any]]
    recommendation: str
