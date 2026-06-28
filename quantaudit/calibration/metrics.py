"""
quantaudit.calibration.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure mathematical functions for evaluating the probabilistic calibration 
of binary classification models.


While standard libraries like scikit-learn provide Log Loss and Brier Score, 
this module focuses on advanced diagnostic metrics that sklearn lacks, 
specifically designed to identify overconfidence and tail-risk miscalibration 
(Overconfidence Index, Tail Error).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import brier_score_loss, log_loss

PROB_EPS = 1e-15

@dataclass
class CalibrationMetrics:
    """Dataclass to hold the results of a calibration evaluation."""
    log_loss: float
    brier_score: float
    ece: float
    oci: float
    tail_error: float
    n_samples: int

def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validates and cleans inputs, removing NaNs from predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Filter out NaNs or infinite values in predictions
    valid_mask = np.isfinite(y_prob)
    y_true_clean = y_true[valid_mask]
    y_prob_clean = y_prob[valid_mask]
    
    # Clip probabilities to avoid log(0)
    y_prob_clean = np.clip(y_prob_clean, PROB_EPS, 1.0 - PROB_EPS)
    
    return y_true_clean, y_prob_clean

def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculates the Expected Calibration Error (ECE).
    
    ECE measures the weighted average of the absolute difference between 
    predicted probabilities and empirical frequencies across bins.
    
    Args:
        y_true: Array of true binary labels (0 or 1).
        y_prob: Array of predicted probabilities.
        n_bins: Number of bins to divide the probability space into.
        
    Returns:
        The ECE score (lower is better, 0.0 is perfect).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float('nan')
        
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
        
    return float(ece)

def overconfidence_index(
    y_true: np.ndarray, y_prob: np.ndarray, high_conf_threshold: float = 0.7
) -> float:
    """
    Calculates the Overconfidence Index (OCI).
    
    Measures the average absolute gap between predicted probability and 
    empirical frequency specifically in the high-confidence zone.
    
    Args:
        y_true: Array of true binary labels.
        y_prob: Array of predicted probabilities.
        high_conf_threshold: Minimum probability to be considered "high confidence".
        
    Returns:
        The OCI score (lower is better). Returns NaN if no samples exceed the threshold.
    """
    high_mask = y_prob > high_conf_threshold
    if not high_mask.any():
        return float('nan')
        
    empirical = y_true[high_mask].mean()
    return float(np.mean(np.abs(y_prob[high_mask] - empirical)))

def tail_error(
    y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10
) -> float:
    """
    Calculates the Tail Error.
    
    Measures the calibration error specifically in the most confident 
    predictions (the top X% of probabilities).
    
    Args:
        y_true: Array of true binary labels.
        y_prob: Array of predicted probabilities.
        top_frac: Fraction of the most confident predictions to evaluate.
        
    Returns:
        The Tail Error score (lower is better). Returns NaN if input is empty.
    """
    if len(y_prob) == 0:
        return float('nan')
        
    cutoff = np.quantile(y_prob, 1.0 - top_frac)
    tail_mask = y_prob >= cutoff
    
    if not tail_mask.any():
        return float('nan')
        
    return float(np.mean(np.abs(y_prob[tail_mask] - y_true[tail_mask])))

def evaluate_model(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10, 
    high_conf_threshold: float = 0.7, 
    top_frac: float = 0.10
) -> CalibrationMetrics:
    """
    Computes all calibration metrics for a single model.
    
    Args:
        y_true: Array of true binary labels.
        y_prob: Array of predicted probabilities.
        n_bins: Number of bins for ECE.
        high_conf_threshold: Threshold for OCI.
        top_frac: Fraction for Tail Error.
        
    Returns:
        A CalibrationMetrics dataclass containing all computed scores.
    """
    y, p = _validate_inputs(y_true, y_prob)
    
    if len(y) == 0:
        return CalibrationMetrics(
            log_loss=float('nan'), brier_score=float('nan'), ece=float('nan'),
            oci=float('nan'), tail_error=float('nan'), n_samples=0
        )

    return CalibrationMetrics(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p, n_bins),
        oci=overconfidence_index(y, p, high_conf_threshold),
        tail_error=tail_error(y, p, top_frac),
        n_samples=int(len(y))
    )