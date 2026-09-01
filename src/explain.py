from typing import List, Dict, Any, Union
import numpy as np


def compute_instance_explanations(
    explainer: Any,
    transformed_features: np.ndarray,
    feature_names: List[str],
    top_k: int = 3,
) -> List[Dict[str, Union[str, float]]]:
    """
    Computes local SHAP explanation values for a single customer instance
    and returns the top K most impactful drivers.
    """
    shap_vals = explainer.shap_values(transformed_features)

    # Handle SHAP output variations (list of classes vs 2D/3D numpy array)
    if isinstance(shap_vals, list):
        # Binary classification list [class_0_vals, class_1_vals]
        class_1_shap = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            # Shape (n_samples, n_features, n_classes)
            class_1_shap = shap_vals[:, :, 1]
        else:
            # Shape (n_samples, n_features) - log-odds or margin output
            class_1_shap = shap_vals
    else:
        raise ValueError(f"Unsupported SHAP output format: {type(shap_vals)}")

    # Extract vector for the single instance
    instance_shap = class_1_shap[0]

    impacts = []
    for feat_name, val in zip(feature_names, instance_shap):
        val_float = float(val)
        desc = (
            "Increases churn risk"
            if val_float > 0
            else "Supports customer retention"
        )
        impacts.append(
            {
                "feature": feat_name,
                "impact_score": round(val_float, 4),
                "description": desc,
            }
        )

    # Sort by absolute impact score descending
    sorted_impacts = sorted(
        impacts, key=lambda x: abs(float(x["impact_score"])), reverse=True
    )
    return sorted_impacts[:top_k]
