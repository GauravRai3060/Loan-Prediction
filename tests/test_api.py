import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True

def test_score_prime_applicant():
    payload = {
        "LoanID": "BM-TEST-PRIME",
        "Age": 45,
        "Income": 95000.0,
        "LoanAmount": 20000.0,
        "CreditScore": 760,
        "MonthsEmployed": 60,
        "NumCreditLines": 2,
        "InterestRate": 6.5,
        "LoanTerm": 36,
        "DTIRatio": 0.22,
        "Education": "Master's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Home",
        "HasCoSigner": "Yes"
    }
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "default_probability" in data
    assert "decision" in data
    assert "risk_rating" in data
    assert data["decision"] in ["APPROVE", "MANUAL_REVIEW"]

def test_score_subprime_applicant():
    payload = {
        "LoanID": "BM-TEST-SUBPRIME",
        "Age": 21,
        "Income": 18000.0,
        "LoanAmount": 85000.0,
        "CreditScore": 420,
        "MonthsEmployed": 3,
        "NumCreditLines": 1,
        "InterestRate": 24.0,
        "LoanTerm": 48,
        "DTIRatio": 0.85,
        "Education": "High School",
        "EmploymentType": "Unemployed",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "No",
        "LoanPurpose": "Auto",
        "HasCoSigner": "No"
    }
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["default_probability"] > 0.30
    assert data["decision"] in ["DECLINE", "MANUAL_REVIEW"]

def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "threshold_analysis" in data

if __name__ == "__main__":
    test_health()
    test_score_prime_applicant()
    test_score_subprime_applicant()
    test_metrics_endpoint()
    print("All API tests passed successfully!")
