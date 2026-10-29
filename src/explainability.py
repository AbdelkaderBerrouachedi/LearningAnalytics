"""
SHAP Explainability and Cross-Presentation Explanation Stability Module.
Evaluates TreeExplainer feature attributions and computes rank correlations across cohorts.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr
import shap

def compute_shap_explanations(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute SHAP values using TreeExplainer on subsampled test cases,
    and evaluate rank stability against training cohort attributions.
    """
    explainer = shap.TreeExplainer(model)
    rng = np.random.RandomState(random_seed)

    # Subsample test cases
    n_test = len(X_test)
    sample_size = min(max_samples, n_test)
    test_idx = rng.choice(n_test, sample_size, replace=False)
    X_test_sample = X_test[test_idx]

    shap_test = explainer.shap_values(X_test_sample)
    if isinstance(shap_test, list):
        shap_test = shap_test[1]  # positive class (withdrawal)

    mean_abs_test = np.mean(np.abs(shap_test), axis=0)

    # Subsample training cases for cross-presentation comparison
    n_train = len(X_train)
    train_size = min(max_samples, n_train)
    train_idx = rng.choice(n_train, train_size, replace=False)
    X_train_sample = X_train[train_idx]

    shap_train = explainer.shap_values(X_train_sample)
    if isinstance(shap_train, list):
        shap_train = shap_train[1]

    mean_abs_train = np.mean(np.abs(shap_train), axis=0)

    # Rank correlations across presentations
    tau, p_tau = kendalltau(mean_abs_train, mean_abs_test)
    rho, p_rho = spearmanr(mean_abs_train, mean_abs_test)
    r_val, p_r = pearsonr(mean_abs_train, mean_abs_test)

    # Sort features by global importance in test set
    ranked_indices = np.argsort(mean_abs_test)[::-1]
    ranked_features = [
        {
            "rank": rank + 1,
            "feature": feature_names[idx],
            "mean_abs_shap_test": float(mean_abs_test[idx]),
            "mean_abs_shap_train": float(mean_abs_train[idx])
        }
        for rank, idx in enumerate(ranked_indices)
    ]

    return {
        "ranked_features": ranked_features,
        "stability": {
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r_val),
            "pearson_p": float(p_r)
        },
        "mean_abs_test": mean_abs_test,
        "mean_abs_train": mean_abs_train,
        "test_shap_matrix": shap_test,
        "test_sample_indices": test_idx
    }

