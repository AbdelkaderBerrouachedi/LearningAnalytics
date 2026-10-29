"""
Evaluation metrics module.
Computes PR-AUC, AUROC, Brier score, Expected Calibration Error (ECE),
advisor contact budgets (Top 5%, 10%, 20%), and 1,000-sample bootstrap confidence intervals.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

def compute_budget_metrics(y_true: np.ndarray, y_prob: np.ndarray, budget_pct: float) -> Tuple[float, float]:
    """Compute Precision and Recall when restricting advisor outreach to top budget_pct of students."""
    n = len(y_true)
    k = max(1, int(np.floor(budget_pct * n)))
    top_indices = np.argsort(y_prob)[::-1][:k]
    flagged_true = y_true[top_indices]
    precision = float(np.mean(flagged_true))
    total_positives = int(np.sum(y_true))
    recall = float(np.sum(flagged_true) / total_positives) if total_positives > 0 else 0.0
    return precision, recall

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all standard evaluation metrics for predicted withdrawal risks."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    pr_auc = float(average_precision_score(y_true, y_prob)) if np.sum(y_true) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob, n_bins=10)

    p5, r5 = compute_budget_metrics(y_true, y_prob, 0.05)
    p10, r10 = compute_budget_metrics(y_true, y_prob, 0.10)
    p20, r20 = compute_budget_metrics(y_true, y_prob, 0.20)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece,
        "p5": p5, "r5": r5,
        "p10": p10, "r10": r10,
        "p20": p20, "r20": r20,
    }

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """Compute 95% empirical bootstrap confidence intervals [mean, low, high] for all metrics."""
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    boot_metrics = {k: [] for k in ["pr_auc", "roc_auc", "brier", "p5", "r5", "p10", "r10", "p20", "r20"]}

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        sample_prob = y_prob[idx]
        if np.sum(sample_y) == 0 or np.sum(sample_y) == n:
            continue
        res = evaluate_predictions(sample_y, sample_prob)
        for k in boot_metrics:
            boot_metrics[k].append(res[k])

    ci_results = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            mu = float(np.mean(vals))
            lo = float(np.percentile(vals, 2.5))
            hi = float(np.percentile(vals, 97.5))
            ci_results[k] = (mu, lo, hi)
        else:
            ci_results[k] = (0.0, 0.0, 0.0)

    return ci_results

