"""
quantaudit.calibration.metrics
Pure mathematical functions for evaluating the probabilistic calibration 
of binary classification models.
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
    if len(y_true[valid_mask]) == 0:
        raise ValueError("No valid observations after removing NaNs or infinite values.")
    y_true_clean = y_true[valid_mask]
    y_prob_clean = y_prob[valid_mask]
    
    # Validate probs are in [0, 1]
    if np.any((y_prob_clean < 0) | (y_prob_clean > 1)):
        raise ValueError("Predicted probabilities must be in the range [0, 1].")
        
    # Validate that y_true contains only binary labels {0, 1}
    valid_labels = np.isin(y_true_clean, [0, 1])
    if not np.all(valid_labels):
        raise ValueError(
            f"y_true must contain only binary labels {{0,1}}. "
            f"Found invalid values: {np.unique(y_true_clean[~valid_labels])}"
        )
    
    # Clip probabilities to avoid log(0)
    y_prob_clean = np.clip(y_prob_clean, PROB_EPS, 1.0 - PROB_EPS)
    
    return y_true_clean, y_prob_clean

def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculates the Expected Calibration Error (ECE)."""
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
    """Calculates the Overconfidence Index (OCI)."""
    high_mask = y_prob > high_conf_threshold
    if not high_mask.any():
        return float('nan')
        
    empirical = y_true[high_mask].mean()
    return float(np.mean(np.abs(y_prob[high_mask] - empirical)))

def tail_error(
    y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10
) -> float:
    """Calculates the Tail Error."""
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
    """Computes all calibration metrics for a single model."""
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