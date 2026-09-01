import pytest
from fastapi.testclient import TestClient
from src.app import app, artifacts
from src.train import train_and_evaluate


@pytest.fixture(scope="session", autouse=True)
def ensure_models_trained():
    # Ensure models are trained and saved before API tests run
    train_and_evaluate(n_samples=1000, save_artifacts=True, random_state=42)
    artifacts.load()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
    assert data["model_loaded"] is True


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "explainable-churn-api"
    assert data["model_loaded"] is True


def test_predict_high_risk_customer(client):
    payload = {
        "tenure_months": 2,
        "monthly_charges": 115.0,
        "total_tickets": 6,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "churn_risk_score" in data
    assert "will_churn" in data
    assert "risk_level" in data
    assert "top_churn_drivers" in data
    
    assert data["churn_risk_score"] >= 0.50
    assert data["will_churn"] is True
    assert data["risk_level"] in ["Medio", "Alto"]
    assert len(data["top_churn_drivers"]) == 3


def test_predict_low_risk_customer(client):
    payload = {
        "tenure_months": 65,
        "monthly_charges": 25.0,
        "total_tickets": 0,
        "contract_type": "two-year",
        "payment_method": "credit_card"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["churn_risk_score"] < 0.40
    assert data["will_churn"] is False
    assert data["risk_level"] in ["Basso", "Medio"]
    assert len(data["top_churn_drivers"]) == 3


def test_predict_invalid_input(client):
    # Invalid contract_type and negative tickets
    invalid_payload = {
        "tenure_months": -5,
        "monthly_charges": -10.0,
        "total_tickets": -1,
        "contract_type": "invalid-type",
        "payment_method": "unknown_method"
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
