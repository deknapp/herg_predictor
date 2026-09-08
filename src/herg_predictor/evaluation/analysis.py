"""Analysis and visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        label: Legend label

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    if label is None:
        label = f"AUROC = {roc_auc:.3f}"
    else:
        label = f"{label} (AUROC = {roc_auc:.3f})"

    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        ax: Matplotlib axes
        label: Legend label

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)

    if label is None:
        label = f"AUPRC = {pr_auc:.3f}"
    else:
        label = f"{label} (AUPRC = {pr_auc:.3f})"

    ax.plot(recall, precision, linewidth=2, label=label)
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1, label=f"Baseline = {baseline:.3f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot calibration curve to assess prediction reliability.

    A well-calibrated model's predicted probabilities should match
    the observed frequencies.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    ax.plot(prob_pred, prob_true, "o-", linewidth=2, markersize=8, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return fig


def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Analyze prediction errors.

    Args:
        df: DataFrame with 'smiles' column and molecular properties
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with DataFrames for different error types:
        - 'false_positives': Predicted positive but actually negative
        - 'false_negatives': Predicted negative but actually positive
        - 'summary': Summary statistics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    df = df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["y_pred_proba"] = y_pred_proba
    df["error_type"] = "correct"

    # Identify error types
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    df.loc[fp_mask, "error_type"] = "false_positive"
    df.loc[fn_mask, "error_type"] = "false_negative"

    # Extract error sets
    false_positives = df[fp_mask].sort_values("y_pred_proba", ascending=False)
    false_negatives = df[fn_mask].sort_values("y_pred_proba", ascending=True)

    # Summary statistics
    summary = pd.DataFrame({
        "category": ["Total", "True Positives", "True Negatives", "False Positives", "False Negatives"],
        "count": [
            len(df),
            ((y_pred == 1) & (y_true == 1)).sum(),
            ((y_pred == 0) & (y_true == 0)).sum(),
            fp_mask.sum(),
            fn_mask.sum(),
        ],
    })
    summary["percentage"] = 100 * summary["count"] / len(df)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "all_predictions": df,
        "summary": summary,
    }


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Prediction Distribution",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot distribution of predicted probabilities by true class.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Separate by class
    probs_neg = y_pred_proba[y_true == 0]
    probs_pos = y_pred_proba[y_true == 1]

    # Plot histograms
    ax.hist(probs_neg, bins=50, alpha=0.5, label="Non-inhibitors", density=True)
    ax.hist(probs_pos, bins=50, alpha=0.5, label="Inhibitors", density=True)

    ax.axvline(x=0.5, color="k", linestyle="--", linewidth=1, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str = "evaluation_report.png",
) -> plt.Figure:
    """
    Create a comprehensive evaluation report figure.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    plot_roc_curve(y_true, y_pred_proba, ax=axes[0, 0])
    plot_precision_recall_curve(y_true, y_pred_proba, ax=axes[0, 1])
    plot_calibration_curve(y_true, y_pred_proba, ax=axes[1, 0])
    plot_prediction_distribution(y_true, y_pred_proba, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
