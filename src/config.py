"""
Configuration module for Bank Muscat Loan Default Prediction Project.
Centralizes paths, features, target definition, and modeling hyperparams.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "Loan_default.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_PATH = BASE_DIR / "artifacts" / "loan_audit.db"

# Model Artifact Paths
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
LOGISTIC_REG_PATH = ARTIFACTS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_PATH = ARTIFACTS_DIR / "random_forest.joblib"
XGBOOST_PATH = ARTIFACTS_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"
SHAP_DATA_PATH = ARTIFACTS_DIR / "shap_values.joblib"
REFERENCE_DATA_PATH = ARTIFACTS_DIR / "reference_baseline.parquet"

# Feature Definitions
TARGET_COL = "Default"
ID_COL = "LoanID"

NUMERICAL_COLS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]

CATEGORICAL_COLS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]

# Engineered Features
ENGINEERED_COLS = [
    "DebtToIncomeTotal",
    "LoanToIncomeRatio",
    "MonthlyPaymentEst",
    "PaymentToIncomeRatio",
    "RiskCreditScoreTier",
    "EmploymentStabilityScore",
    "InterestBurdenScore"
]

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# Thresholds & Business Rules (Bank Muscat Underwriting)
DEFAULT_PROB_THRESHOLD = 0.25 # Optimized threshold to reduce NPA by >=30%
TARGET_AUC = 0.85
NPA_REDUCTION_TARGET_PCT = 30.0
