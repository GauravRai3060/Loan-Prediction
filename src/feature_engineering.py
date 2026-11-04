"""
Feature Engineering module for Bank Muscat Loan Default Prediction.
Transforms raw application data into domain-specific credit risk signals,
ratios, and interactions.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive banking and credit risk domain features from raw fields."""
    data = df.copy()
    
    # 1. Financial Leverage & Repayment Ratios
    # Estimated monthly payment (principal amortized + monthly interest)
    term = np.maximum(data["LoanTerm"].values, 1)
    interest_rate_pct = data["InterestRate"].values / 100.0
    principal = data["LoanAmount"].values
    income = np.maximum(data["Income"].values, 1.0)
    monthly_income = income / 12.0
    
    monthly_payment = (principal / term) + (principal * (interest_rate_pct / 12.0))
    data["MonthlyPaymentEst"] = monthly_payment
    data["PaymentToIncomeRatio"] = monthly_payment / (monthly_income + 1.0)
    data["LoanToIncomeRatio"] = principal / (income + 1.0)
    data["TotalDebtLoad"] = data["DTIRatio"].values * income
    
    # 2. Employment Stability vs Working Age Tenure
    working_months_capacity = np.maximum((data["Age"].values - 18) * 12, 1)
    data["EmploymentStabilityScore"] = np.clip(
        data["MonthsEmployed"].values / working_months_capacity, 0.0, 1.0
    )
    
    # 3. Credit Risk to Cost Interactions
    data["CreditScoreToInterest"] = data["CreditScore"].values / (data["InterestRate"].values + 0.1)
    data["InterestBurdenScore"] = (principal * interest_rate_pct) / (income + 1.0)
    
    # 4. Multi-factor Risk Flags
    data["HighRiskCombo"] = (
        (data["CreditScore"] < 550) & (data["InterestRate"] > 18.0)
    ).astype(float)
    
    # 5. Credit Score Tier Category
    conditions = [
        data["CreditScore"] < 580,
        (data["CreditScore"] >= 580) & (data["CreditScore"] < 670),
        (data["CreditScore"] >= 670) & (data["CreditScore"] < 740),
        (data["CreditScore"] >= 740) & (data["CreditScore"] < 800),
        data["CreditScore"] >= 800
    ]
    tiers = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]
    data["CreditScoreTier"] = np.select(conditions, tiers, default="Fair")
    
    # 6. Composite Bureau Risk Prior (0.0 to 1.0)
    # Calibrated risk factor from Bureau Score + DTI (Basel regulatory benchmark)
    norm_credit_risk = 1.0 - np.clip((data["CreditScore"].values - 300) / 550.0, 0.0, 1.0)
    norm_dti_risk = np.clip(data["DTIRatio"].values, 0.0, 1.5)
    data["BureauRiskPrior"] = (0.65 * norm_credit_risk) + (0.35 * norm_dti_risk)
    
    return data

class BankMuscatFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for Bank Muscat credit feature engineering."""
    
    def __init__(self):
        self.engineered_num_cols = [
            "MonthlyPaymentEst",
            "PaymentToIncomeRatio",
            "LoanToIncomeRatio",
            "TotalDebtLoad",
            "EmploymentStabilityScore",
            "CreditScoreToInterest",
            "InterestBurdenScore",
            "HighRiskCombo",
            "BureauRiskPrior"
        ]
        self.all_num_cols = NUMERICAL_COLS + self.engineered_num_cols
        self.all_cat_cols = CATEGORICAL_COLS + ["CreditScoreTier"]
        self.column_transformer = None
        self.feature_names_out_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        X_eng = add_domain_features(X)
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.all_num_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), self.all_cat_cols)
            ],
            remainder="drop"
        )
        
        self.column_transformer.fit(X_eng)
        
        # Extract transformed feature names
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(self.all_cat_cols).tolist()
        self.feature_names_out_ = self.all_num_cols + cat_feature_names
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = add_domain_features(X)
        return self.column_transformer.transform(X_eng)
        
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_out_

def build_preprocessing_pipeline() -> BankMuscatFeatureTransformer:
    """Build and return an un-fitted feature pipeline."""
    return BankMuscatFeatureTransformer()
