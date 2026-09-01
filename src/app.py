from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
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


# In-memory artifact holder
class ArtifactStore:
    def __init__(self):
        self.classifier_bundle: Optional[Dict[str, Any]] = None
        self.explainer: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self.classifier_bundle is not None and self.explainer is not None

    def load(self, classifier_path: Path = CLASSIFIER_PATH, explainer_path: Path = EXPLAINER_PATH):
        if not classifier_path.exists() or not explainer_path.exists():
            raise FileNotFoundError(
                f"Modelli non trovati. Assicurati che esistano {classifier_path} e {explainer_path}. "
                "Esegui prima `python src/train.py`."
            )
        self.classifier_bundle = joblib.load(classifier_path)
        self.explainer = joblib.load(explainer_path)


artifacts = ArtifactStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML & xAI models
    try:
        artifacts.load()
        print("Artefatti ML e Explainer SHAP caricati con successo.")
    except Exception as e:
        print(f"ATTENZIONE: Impossibile caricare gli artefatti all'avvio: {e}")
    yield
    # Shutdown logic (if any)


app = FastAPI(
    title="Explainable Churn Prediction API",
    version="1.0.0",
    description="Microservizio ML end-to-end per la predizione del churn con Explainable AI (SHAP TreeExplainer).",
    lifespan=lifespan,
)


# --- Pydantic Schemas ---

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(
        ...,
        ge=0,
        description="Mesi di permanenza del cliente con il servizio",
        examples=[12],
    )
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        description="Spesa mensile del cliente in EUR/USD",
        examples=[75.50],
    )
    total_tickets: int = Field(
        ...,
        ge=0,
        description="Numero totale di ticket di supporto aperti",
        examples=[3],
    )
    contract_type: str = Field(
        ...,
        description="Tipo di contratto (es. 'month-to-month', 'one-year', 'two-year')",
        examples=["month-to-month"],
    )
    payment_method: str = Field(
        ...,
        description="Metodo di pagamento (es. 'electronic_check', 'bank_transfer', 'credit_card')",
        examples=["electronic_check"],
    )

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in CONTRACT_TYPES:
            raise ValueError(f"contract_type deve essere uno tra: {CONTRACT_TYPES}")
        return clean_v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in PAYMENT_METHODS:
            raise ValueError(f"payment_method deve essere uno tra: {PAYMENT_METHODS}")
        return clean_v


class FeatureImpact(BaseModel):
    feature: str = Field(..., description="Nome della feature trasformata")
    impact_score: float = Field(..., description="SHAP value (impatto locale)")
    description: str = Field(..., description="Spiegazione qualitativa dell'impatto")


class PredictionResponse(BaseModel):
    churn_risk_score: float = Field(..., description="Probabilità predetta di churn (0.0 - 1.0)")
    will_churn: bool = Field(..., description="Predizione binaria (True se rischio >= soglia)")
    risk_level: str = Field(..., description="Livello di rischio categorico ('Basso', 'Medio', 'Alto')")
    top_churn_drivers: List[FeatureImpact] = Field(
        ..., description="Top feature con il maggiore impatto positivo/negativo sul punteggio"
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool


# --- API Routes ---

@app.get("/", summary="Root Info")
def root_info():
    return {
        "service": "Explainable Customer Retention & Churn API",
        "status": "online",
        "docs": "/docs",
        "model_loaded": artifacts.is_loaded,
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health():
    return {
        "status": "ok" if artifacts.is_loaded else "degraded",
        "service": "explainable-churn-api",
        "version": "1.0.0",
        "model_loaded": artifacts.is_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict Churn & Explain")
def predict_and_explain(customer: CustomerFeatures):
    if not artifacts.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artefatti ML non caricati. Eseguire prima il training.",
        )

    # Convert customer data to DataFrame
    input_dict = customer.model_dump()
    input_df = pd.DataFrame([input_dict])

    try:
        # Preprocessing
        preprocessor = artifacts.classifier_bundle["preprocessor"]
        model = artifacts.classifier_bundle["model"]
        feature_names = artifacts.classifier_bundle["feature_names"]

        X_trans = preprocessor.transform(input_df)

        # Predict probability
        proba = float(model.predict_proba(X_trans)[0, 1])

        # Compute SHAP explanations
        top_drivers = compute_instance_explanations(
            explainer=artifacts.explainer,
            transformed_features=X_trans,
            feature_names=feature_names,
            top_k=3,
        )

        # Risk level determination
        if proba >= HIGH_RISK_THRESHOLD:
            risk_level = "Alto"
        elif proba >= MEDIUM_RISK_THRESHOLD:
            risk_level = "Medio"
        else:
            risk_level = "Basso"

        return {
            "churn_risk_score": round(proba, 4),
            "will_churn": proba >= DECISION_THRESHOLD,
            "risk_level": risk_level,
            "top_churn_drivers": top_drivers,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'inferenza o il calcolo SHAP: {str(e)}",
        )
