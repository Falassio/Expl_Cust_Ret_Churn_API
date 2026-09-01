from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
EXPLAINER_PATH = MODELS_DIR / "explainer.joblib"
DATASET_PATH = DATA_DIR / "telco_churn.csv"

# Feature definitions
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_tickets"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churn"

# Allowed categorical values for validation & metadata
CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
PAYMENT_METHODS = ["electronic_check", "bank_transfer", "credit_card"]

# Risk score thresholds
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.35
DECISION_THRESHOLD = 0.50

# Server configurations
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
