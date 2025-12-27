"""Classification metrics for model evaluation."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        # Threshold-independent
        "auroc": roc_auc_score(y_true, y_pred_proba),
        "auprc": average_precision_score(y_true, y_pred_proba),
        
        # Threshold-dependent
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),  # True positive rate
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # True negative rate
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        
        # Raw counts
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        
        # Additional
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
    }
    
    return metrics


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute metrics with bootstrap confidence intervals.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        seed: Random seed
        
    Returns:
        Dictionary with 'mean', 'lower', 'upper' for each metric
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    
    # Metrics to bootstrap
    metric_names = ["auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc"]
    bootstrap_results = {name: [] for name in metric_names}
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        
        # Skip if only one class present
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        metrics = compute_classification_metrics(y_true_boot, y_pred_boot, threshold)
        for name in metric_names:
            bootstrap_results[name].append(metrics[name])
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    results = {}
    
    for name in metric_names:
        values = np.array(bootstrap_results[name])
        results[name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "lower": np.percentile(values, 100 * alpha / 2),
            "upper": np.percentile(values, 100 * (1 - alpha / 2)),
        }
    
    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Find optimal classification threshold for a given metric.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden')
        thresholds: Thresholds to try (default: 0.01 to 0.99)
        
    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    best_threshold = 0.5
    best_value = 0.0
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J statistic = sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = sensitivity + specificity - 1
        elif metric == "mcc":
            value = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value
