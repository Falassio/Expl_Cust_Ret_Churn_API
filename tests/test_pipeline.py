import pytest
import numpy as np
import pandas as pd
from src.data_pipeline import (
    generate_synthetic_crm_data,
    build_preprocessor,
    extract_feature_names,
    prepare_data_splits,
)
from src.train import train_and_evaluate
from src.explain import compute_instance_explanations


def test_synthetic_data_generation():
    """Verifies synthetic dataset generation schema and statistical integrity."""
    df = generate_synthetic_crm_data(n_samples=500, random_seed=42)
    assert len(df) == 500
    assert "churn" in df.columns
    assert set(df["churn"].unique()).issubset({0, 1})
    expected_cols = [
        "tenure_months",
        "monthly_charges",
        "total_tickets",
        "contract_type",
        "payment_method",
    ]
    assert all(col in df.columns for col in expected_cols)


def test_preprocessor_transformation():
    """Verifies ColumnTransformer scales numericals and encodes categoricals correctly."""
    df = generate_synthetic_crm_data(n_samples=100)
    X = df.drop(columns=["churn"])
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    feature_names = extract_feature_names(preprocessor)

    assert X_trans.shape[0] == 100
    assert X_trans.shape[1] == len(feature_names)
    assert len(feature_names) >= 5


def test_train_and_evaluate():
    """Verifies end-to-end model training runs and meets baseline performance benchmarks."""
    results = train_and_evaluate(n_samples=500, save_artifacts=False, random_state=42)

    assert "model" in results
    assert "preprocessor" in results
    assert "explainer" in results
    assert results["roc_auc"] > 0.70
    assert results["accuracy"] > 0.65


def test_explainability_computation():
    """Verifies SHAP TreeExplainer computes valid top feature impacts and interpretations."""
    results = train_and_evaluate(n_samples=300, save_artifacts=False, random_state=42)
    preprocessor = results["preprocessor"]
    explainer = results["explainer"]
    feature_names = results["feature_names"]

    sample_df = pd.DataFrame(
        [
            {
                "tenure_months": 2,
                "monthly_charges": 110.0,
                "total_tickets": 5,
                "contract_type": "month-to-month",
                "payment_method": "electronic_check",
            }
        ]
    )

    X_trans = preprocessor.transform(sample_df)
    explanations = compute_instance_explanations(
        explainer, X_trans, feature_names, top_k=3
    )

    assert len(explanations) == 3
    assert all(
        "feature" in item and "impact_score" in item and "description" in item
        for item in explanations
    )
    assert isinstance(explanations[0]["impact_score"], float)
