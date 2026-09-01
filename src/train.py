from pathlib import Path
import sys

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import shap

from src.config import (
    MODELS_DIR,
    CLASSIFIER_PATH,
    EXPLAINER_PATH,
    DATASET_PATH,
)
from src.data_pipeline import (
    load_or_generate_data,
    build_preprocessor,
    extract_feature_names,
    prepare_data_splits,
)


def train_and_evaluate(
    n_samples: int = 5000,
    save_artifacts: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Executes end-to-end training pipeline:
    1. Loads / generates dataset
    2. Fits ColumnTransformer preprocessor
    3. Trains LightGBM classifier
    4. Computes evaluation metrics
    5. Builds SHAP TreeExplainer
    6. Serializes artifacts to disk
    """
    print("1. Loading / generating dataset...")
    df = load_or_generate_data(DATASET_PATH)
    print(f"   Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"   Churn distribution: {df['churn'].value_counts(normalize=True).to_dict()}")

    print("2. Splitting Train/Test & Preprocessing...")
    X_train, X_test, y_train, y_test = prepare_data_splits(
        df, test_size=0.2, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = extract_feature_names(preprocessor)
    print(f"   Transformed features ({len(feature_names)}): {feature_names}")

    print("3. Training LightGBM Classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state,
        verbose=-1,
    )
    clf.fit(X_train_trans, y_train)

    print("4. Evaluating Model Performance...")
    y_pred_proba = clf.predict_proba(X_test_trans)[:, 1]
    y_pred = clf.predict(X_test_trans)

    auc_score = roc_auc_score(y_test, y_pred_proba)
    acc_score = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("   --- Model Performance Metrics ---")
    print(f"   ROC-AUC: {auc_score:.4f}")
    print(f"   Accuracy: {acc_score:.4f}")
    print(classification_report(y_test, y_pred))

    print("5. Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        classifier_data = {
            "preprocessor": preprocessor,
            "model": clf,
            "feature_names": feature_names,
            "metrics": {
                "roc_auc": round(float(auc_score), 4),
                "accuracy": round(float(acc_score), 4),
            },
        }
        joblib.dump(classifier_data, CLASSIFIER_PATH)
        joblib.dump(explainer, EXPLAINER_PATH)
        print(f"   Artifacts successfully persisted to:\n   - {CLASSIFIER_PATH}\n   - {EXPLAINER_PATH}")

    return {
        "model": clf,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "feature_names": feature_names,
        "roc_auc": auc_score,
        "accuracy": acc_score,
        "report": report,
    }


if __name__ == "__main__":
    train_and_evaluate()
