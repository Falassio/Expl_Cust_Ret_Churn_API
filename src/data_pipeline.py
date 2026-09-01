from pathlib import Path
import sys
from typing import Tuple, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    DATASET_PATH,
)


def generate_synthetic_crm_data(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic CRM / Telco churn data."""
    np.random.seed(random_seed)
    tenure_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_tickets = np.random.poisson(lam=1.5, size=n_samples)
    contract_type = np.random.choice(
        ["month-to-month", "one-year", "two-year"],
        p=[0.55, 0.25, 0.20],
        size=n_samples,
    )
    payment_method = np.random.choice(
        ["electronic_check", "bank_transfer", "credit_card"],
        p=[0.40, 0.30, 0.30],
        size=n_samples,
    )

    # Churn probability calculation (realistic business log-odds)
    raw_score = (
        0.04 * monthly_charges
        - 0.05 * tenure_months
        + 0.60 * total_tickets
        + (contract_type == "month-to-month") * 1.2
        + (payment_method == "electronic_check") * 0.4
        - 2.0
    )
    prob = 1.0 / (1.0 + np.exp(-raw_score))
    churn = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": np.round(monthly_charges, 2),
            "total_tickets": total_tickets,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "churn": churn,
        }
    )
    return df


def load_or_generate_data(csv_path: Optional[Path] = None, save_if_generated: bool = True) -> pd.DataFrame:
    """Loads existing dataset from CSV or generates synthetic data if not found."""
    target_path = csv_path or DATASET_PATH
    if target_path.exists():
        df = pd.read_csv(target_path)
    else:
        df = generate_synthetic_crm_data()
        if save_if_generated:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target_path, index=False)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def extract_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts output feature names after ColumnTransformer fitting."""
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_feature_names


def prepare_data_splits(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into stratified train and test sets."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
