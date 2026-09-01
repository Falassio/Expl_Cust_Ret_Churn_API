from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.config import (
    CLASSIFIER_PATH,
    EXPLAINER_PATH,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    DECISION_THRESHOLD,
    CONTRACT_TYPES,
    PAYMENT_METHODS,
)
from src.explain import compute_instance_explanations

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("churn-api")


class ArtifactStore:
    """Thread-safe in-memory cache for ML models and SHAP explainers."""
    def __init__(self):
        self.classifier_bundle: Optional[Dict[str, Any]] = None
        self.explainer: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self.classifier_bundle is not None and self.explainer is not None

    def load(self, classifier_path: Path = CLASSIFIER_PATH, explainer_path: Path = EXPLAINER_PATH):
        if not classifier_path.exists() or not explainer_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {classifier_path} and {explainer_path}. "
                "Please run `python src/train.py` first."
            )
        self.classifier_bundle = joblib.load(classifier_path)
        self.explainer = joblib.load(explainer_path)


artifacts = ArtifactStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to handle startup artifact loading and graceful shutdown."""
    try:
        artifacts.load()
        logger.info("Machine learning model and SHAP explainer loaded successfully.")
    except Exception as e:
        logger.warning(f"Unable to load artifacts during startup: {e}")
    yield


app = FastAPI(
    title="Explainable Customer Retention & Churn API",
    version="1.0.0",
    description=(
        "Production-grade Machine Learning microservice for customer churn prediction "
        "with local feature attribution powered by SHAP TreeExplainer."
    ),
    lifespan=lifespan,
)


# --- Pydantic Data Contracts ---

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(
        ...,
        ge=0,
        description="Customer account tenure in months",
        examples=[12],
    )
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        description="Monthly subscription charge in USD/EUR",
        examples=[75.50],
    )
    total_tickets: int = Field(
        ...,
        ge=0,
        description="Total customer support tickets opened to date",
        examples=[3],
    )
    contract_type: str = Field(
        ...,
        description="Subscription contract tier ('month-to-month', 'one-year', 'two-year')",
        examples=["month-to-month"],
    )
    payment_method: str = Field(
        ...,
        description="Active billing method ('electronic_check', 'bank_transfer', 'credit_card')",
        examples=["electronic_check"],
    )

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in CONTRACT_TYPES:
            raise ValueError(f"contract_type must be one of: {CONTRACT_TYPES}")
        return clean_v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in PAYMENT_METHODS:
            raise ValueError(f"payment_method must be one of: {PAYMENT_METHODS}")
        return clean_v


class FeatureImpact(BaseModel):
    feature: str = Field(..., description="Name of the transformed feature")
    impact_score: float = Field(..., description="SHAP local attribution score")
    description: str = Field(..., description="Human-readable business interpretation of the impact")


class PredictionResponse(BaseModel):
    churn_risk_score: float = Field(..., description="Predicted churn probability between 0.0 and 1.0")
    will_churn: bool = Field(..., description="Binary classification outcome based on decision threshold")
    risk_level: str = Field(..., description="Categorical risk tier: 'Low', 'Medium', or 'High'")
    top_churn_drivers: List[FeatureImpact] = Field(
        ..., description="Top feature drivers ranked by absolute contribution to the churn score"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status ('ok' or 'degraded')")
    service: str = Field(..., description="Service identifier name")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Indicates if ML models are loaded and ready for inference")


# --- API Routes ---

@app.get("/", summary="Root Metadata")
def root_info():
    """Returns basic service information and documentation link."""
    return {
        "service": "Explainable Customer Retention & Churn API",
        "status": "online",
        "docs": "/docs",
        "model_loaded": artifacts.is_loaded,
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health():
    """Healthcheck endpoint for orchestrators (Dokploy, Docker, Kubernetes)."""
    return {
        "status": "ok" if artifacts.is_loaded else "degraded",
        "service": "explainable-churn-api",
        "version": "1.0.0",
        "model_loaded": artifacts.is_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict Churn & Explain")
def predict_and_explain(customer: CustomerFeatures):
    """
    Computes churn probability and provides local SHAP explanations for the customer profile.
    """
    if not artifacts.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Please run the training pipeline first.",
        )

    # Convert customer input to DataFrame for transformer pipeline
    input_dict = customer.model_dump()
    input_df = pd.DataFrame([input_dict])

    try:
        preprocessor = artifacts.classifier_bundle["preprocessor"]
        model = artifacts.classifier_bundle["model"]
        feature_names = artifacts.classifier_bundle["feature_names"]

        # Feature transformation
        X_trans = preprocessor.transform(input_df)

        # Model inference (probability of churn)
        proba = float(model.predict_proba(X_trans)[0, 1])

        # Compute instance-level SHAP values
        top_drivers = compute_instance_explanations(
            explainer=artifacts.explainer,
            transformed_features=X_trans,
            feature_names=feature_names,
            top_k=3,
        )

        # Determine risk tier
        if proba >= HIGH_RISK_THRESHOLD:
            risk_level = "High"
        elif proba >= MEDIUM_RISK_THRESHOLD:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "churn_risk_score": round(proba, 4),
            "will_churn": proba >= DECISION_THRESHOLD,
            "risk_level": risk_level,
            "top_churn_drivers": top_drivers,
        }

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred during inference or SHAP computation: {str(e)}",
        )
