"""
quantaudit.calibration.audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Orchestrator for probabilistic calibration audits.

Provides the `CalibrationAuditor` class to evaluate, compare, and visualize 
the calibration of multiple binary classification models simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, Optional

from .metrics import evaluate_model, CalibrationMetrics, _validate_inputs

ArrayLike = Union[np.ndarray, pd.Series, list]

class CalibrationAuditor:
    """
    A class to audit and compare the calibration of probabilistic models.
    
    Attributes:
        y_true: Array of true binary labels.
        model_probs: Dictionary mapping model names to their predicted probabilities.
        n_bins: Number of bins for calibration metrics.
        high_conf_threshold: Threshold for Overconfidence Index.
        top_frac: Fraction of top predictions for Tail Error.
    """

    def __init__(
        self, 
        y_true: ArrayLike, 
        model_probs: Dict[str, ArrayLike],
        n_bins: int = 10,
        high_conf_threshold: float = 0.7,
        top_frac: float = 0.10
    ):
        self.y_true = np.asarray(y_true)
        self.model_probs = {name: np.asarray(probs) for name, probs in model_probs.items()}
        self.n_bins = n_bins
        self.high_conf_threshold = high_conf_threshold
        self.top_frac = top_frac
        
        self._validate_inputs()
        self._metrics_cache: Optional[pd.DataFrame] = None

    def _validate_inputs(self):
        """Validates that inputs are correctly shaped and formatted."""
        if not self.model_probs:
            raise ValueError("model_probs dictionary cannot be empty.")
            
        for name, probs in self.model_probs.items():
            if probs.shape != self.y_true.shape:
                raise ValueError(
                    f"Shape mismatch for model '{name}': "
                    f"y_true {self.y_true.shape} vs probs {probs.shape}"
                )
        if not (0 < self.self.top_frac <= 1):
            raise ValueError("top_frac must be in the range (0, 1].")

    def compute_metrics(self) -> pd.DataFrame:
        """
        Computes all calibration metrics for all provided models.
        
        Returns:
            A pandas DataFrame where rows are models and columns are metrics.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        results = []
        for name, probs in self.model_probs.items():
            metrics = evaluate_model(
                self.y_true, probs, self.n_bins, self.high_conf_threshold, self.top_frac
            )
            row = {"model": name, **metrics.__dict__}
            results.append(row)
            
        self._metrics_cache = pd.DataFrame(results).set_index("model")
        return self._metrics_cache

    def get_reliability_data(self) -> pd.DataFrame:
        """
        Calculates binned reliability data (mean predicted vs empirical rate) 
        for all models.
        
        Returns:
            A pandas DataFrame containing bin centers, predicted means, 
            empirical rates, and counts for each model.
        """
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        all_data = []
        
        for name, probs in self.model_probs.items():
            y, p = _validate_inputs(self.y_true, probs)
            bin_ids = np.digitize(p, bins, right=True) - 1
            bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)
            
            for b in range(self.n_bins):
                mask = bin_ids == b
                count = int(mask.sum())
                
                if count > 0:
                    mean_p = float(p[mask].mean())
                    emp_rate = float(y[mask].mean())
                else:
                    mean_p = np.nan
                    emp_rate = np.nan
                    
                all_data.append({
                    "model": name,
                    "bin": b,
                    "bin_center": (bins[b] + bins[b + 1]) / 2,
                    "mean_predicted": mean_p,
                    "empirical_rate": emp_rate,
                    "count": count,
                    "gap": abs(mean_p - emp_rate) if count > 0 else np.nan
                })
                
        return pd.DataFrame(all_data)

    def plot_reliability_curves(
        self, save_path: Optional[Union[str, Path]] = None, show: bool = True
    ) -> None:
        """
        Plots the reliability curves (calibration plots) for all models.
        
        Args:
            save_path: Optional path to save the figure.
            show: Whether to display the plot using plt.show().
        """
        rel_data = self.get_reliability_data()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        
        for name, group in rel_data.groupby("model"):
            valid = group["count"] > 0
            if valid.any():
                ax.plot(
                    group.loc[valid, "mean_predicted"],
                    group.loc[valid, "empirical_rate"],
                    marker="o", linewidth=2, label=name
                )
                
        ax.set_title("Reliability Curves (Calibration Plots)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Empirical Frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_overconfidence_by_bin(
        self, save_path: Optional[Union[str, Path]] = None, show: bool = True
    ) -> None:
        """
        Plots the calibration gap (|Predicted - Empirical|) by confidence bin 
        for all models.
        
        Args:
            save_path: Optional path to save the figure.
            show: Whether to display the plot using plt.show().
        """
        rel_data = self.get_reliability_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(self.n_bins)
        width = 0.8 / max(len(self.model_probs), 1)
        
        for i, (name, group) in enumerate(rel_data.groupby("model")):
            gaps = group["gap"].fillna(0.0)
            ax.bar(x + i * width, gaps, width=width, label=name)
            
        ax.set_title("Calibration Gap by Confidence Bin")
        ax.set_xlabel("Probability Bin")
        ax.set_ylabel("Absolute Calibration Gap")
        ax.set_xticks(x + width * (len(self.model_probs) - 1) / 2)
        ax.set_xticklabels([f"Bin {i}" for i in range(self.n_bins)])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def summary(self) -> str:
        """
        Generates a formatted text summary of the calibration audit.
        
        Returns:
            A string containing the formatted summary.
        """
        metrics_df = self.compute_metrics()
        
        lines = ["=" * 60]
        lines.append("QUANTITATIVE CALIBRATION AUDIT REPORT")
        lines.append("=" * 60)
        lines.append(f"Models evaluated: {len(metrics_df)}")
        lines.append(f"Total samples: {metrics_df['n_samples'].iloc[0]}")
        lines.append("")
        
        lines.append("METRICS SUMMARY (Lower is better for all metrics)")
        lines.append("-" * 60)
        lines.append(metrics_df.to_string(float_format="%.4f"))
        lines.append("")
        
        # Find best models
        best_ll = metrics_df["log_loss"].idxmin()
        best_ece = metrics_df["ece"].idxmin()
        best_oci = metrics_df["oci"].idxmin()
        
        lines.append("DIAGNOSTICS:")
        lines.append(f"  - Best Log Loss: {best_ll}")
        lines.append(f"  - Best ECE (Global Calibration): {best_ece}")
        lines.append(f"  - Best OCI (High-Confidence Calibration): {best_oci}")
        lines.append("=" * 60)
        
        return "\n".join(lines)