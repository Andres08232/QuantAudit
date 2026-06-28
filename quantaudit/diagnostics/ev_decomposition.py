"""
quantaudit.diagnostics.ev_decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-mortem diagnostic tool for Expected Value (EV) based selection strategies.

This module does NOT predict outcomes or beat the market. It is an "autopsy" 
tool designed to answer a specific question: 
"My model claims I have positive Expected Value (+EV), but my realized ROI 
is negative. Why?"

It decomposes the EV into realized financial outcomes, analyzes threshold 
sensitivities, and measures the divergence between model confidence and 
market pricing to identify if the "edge" is genuine signal or mere model 
overconfidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, Optional, List

ArrayLike = Union[np.ndarray, pd.Series, list]

class EVDecompositionAuditor:
    """
    Audits the realization of Expected Value (EV) in asymmetric payoff scenarios 
    (e.g., sports betting, algorithmic trading, credit risk).
    
    Attributes:
        y_true: Array of actual binary outcomes (1 = win/success, 0 = loss/failure).
        model_prob: Array of predicted probabilities from the model.
        market_price: Array of market prices (e.g., decimal odds). 
                      If not provided, market_prob must be provided.
        market_prob: Array of market implied probabilities. 
                     Calculated from market_price if not provided.
        stake: Fixed monetary stake per event.
    """

    def __init__(
        self,
        y_true: ArrayLike,
        model_prob: ArrayLike,
        market_price: Optional[ArrayLike] = None,
        market_prob: Optional[ArrayLike] = None,
        stake: float = 1.0
    ):
        self.y_true = np.asarray(y_true, dtype=int)
        self.model_prob = np.asarray(model_prob, dtype=float)
        self.stake = float(stake)
        
        if market_price is not None:
            self.market_price = np.asarray(market_price, dtype=float)
            self.market_prob = 1.0 / self.market_price
        elif market_prob is not None:
            self.market_prob = np.asarray(market_prob, dtype=float)
            self.market_price = 1.0 / self.market_prob
        else:
            raise ValueError("Either market_price or market_prob must be provided.")

        self._validate_inputs()
        self._df: Optional[pd.DataFrame] = None

    def _validate_inputs(self):
        """Validates shapes and value ranges."""
        n = len(self.y_true)
        if not (len(self.model_prob) == len(self.market_prob) == n):
            raise ValueError("y_true, model_prob, and market_price/prob must have the same length.")
        
        if not np.all((self.model_prob >= 0) & (self.model_prob <= 1)):
            raise ValueError("model_prob must be between 0 and 1.")
        if not np.all((self.market_prob > 0) & (self.market_prob < 1)):
            raise ValueError("market_prob must be > 0 and < 1 (implied by valid market_price > 1).")
        if not np.all(np.isin(self.y_true, [0, 1])):
            raise ValueError("y_true must contain only 0s and 1s.")

    def _build_base_dataframe(self) -> pd.DataFrame:
        """Calculates core financial and probabilistic metrics."""
        if self._df is not None:
            return self._df

        df = pd.DataFrame({
            "y_true": self.y_true,
            "model_prob": self.model_prob,
            "market_prob": self.market_prob,
            "market_price": self.market_price
        })

        # Core Metrics
        df["edge"] = df["model_prob"] - df["market_prob"]
        df["ev_multiplier"] = df["model_prob"] * df["market_price"]
        df["expected_profit_per_unit"] = df["ev_multiplier"] - 1.0
        df["market_distance"] = df["edge"].abs()

        # Financial Realization
        df["profit"] = np.where(
            df["y_true"] == 1, 
            self.stake * (df["market_price"] - 1.0), 
            -self.stake
        )
        df["realized_roi_pct"] = (df["profit"] / self.stake) * 100.0
        
        # Diagnostic Flags
        df["true_positive_ev"] = df["profit"] > 0
        df["model_claims_value"] = df["edge"] > 0

        self._df = df
        return self._df

    def ev_vs_realized_roi(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Bins events by EV multiplier and calculates realized ROI per bin.
        
        Args:
            n_bins: Number of quantile bins to create.
            
        Returns:
            DataFrame with EV bin statistics.
        """
        df = self._build_base_dataframe()
        
        try:
            df["ev_bin"] = pd.qcut(df["ev_multiplier"], q=n_bins, duplicates="drop", labels=False)
        except ValueError:
            # Fallback if not enough unique values for qcut
            df["ev_bin"] = pd.cut(df["ev_multiplier"], bins=n_bins, labels=False)

        rows = []
        for b, g in df.groupby("ev_bin", observed=True):
            n = len(g)
            rows.append({
                "ev_bin": int(b),
                "mean_ev_multiplier": float(g["ev_multiplier"].mean()),
                "mean_expected_profit": float(g["expected_profit_per_unit"].mean()),
                "realized_roi_pct": float(g["profit"].sum() / (n * self.stake) * 100.0),
                "win_rate_pct": float(g["y_true"].mean() * 100.0),
                "false_positive_rate_pct": float((~g["true_positive_ev"]).mean() * 100.0),
                "n_events": n
            })
        return pd.DataFrame(rows)

    def threshold_sensitivity(self, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Evaluates ROI and false positive rates at specific EV multiplier thresholds.
        
        Args:
            thresholds: List of EV multiplier cutoffs (e.g., [1.02, 1.05]).
            
        Returns:
            DataFrame with performance metrics for each threshold.
        """
        if thresholds is None:
            thresholds = [1.00, 1.02, 1.05, 1.10]
            
        df = self._build_base_dataframe()
        rows = []
        
        for thr in thresholds:
            flagged = df[df["ev_multiplier"] > thr]
            n = len(flagged)
            
            if n == 0:
                rows.append({"ev_threshold": thr, "n_events": 0, "realized_roi_pct": 0.0, 
                             "false_positive_rate_pct": 0.0, "mean_edge_pct": 0.0})
                continue
                
            false_pos = flagged[~flagged["true_positive_ev"]]
            
            rows.append({
                "ev_threshold": thr,
                "n_events": n,
                "realized_roi_pct": float(flagged["profit"].sum() / (n * self.stake) * 100.0),
                "win_rate_pct": float(flagged["y_true"].mean() * 100.0),
                "false_positive_rate_pct": float(len(false_pos) / n * 100.0),
                "mean_edge_pct": float(flagged["edge"].mean() * 100.0)
            })
        return pd.DataFrame(rows)

    def market_distance_analysis(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Analyzes ROI based on the absolute distance between model and market probabilities.
        Helps identify if large disagreements are destructive.
        """
        df = self._build_base_dataframe()
        
        try:
            df["dist_bin"] = pd.qcut(df["market_distance"], q=n_bins, duplicates="drop", labels=False)
        except ValueError:
            df["dist_bin"] = pd.cut(df["market_distance"], bins=n_bins, labels=False)

        rows = []
        for b, g in df.groupby("dist_bin", observed=True):
            n = len(g)
            rows.append({
                "distance_bin": int(b),
                "mean_distance_pct": float(g["market_distance"].mean() * 100.0),
                "realized_roi_pct": float(g["profit"].sum() / (n * self.stake) * 100.0),
                "win_rate_pct": float(g["y_true"].mean() * 100.0),
                "n_events": n
            })
        return pd.DataFrame(rows)

    def compute_correlations(self) -> Dict[str, float]:
        """Computes correlations between expected metrics and realized profit."""
        df = self._build_base_dataframe()
        return {
            "corr_ev_multiplier_vs_profit": float(df["ev_multiplier"].corr(df["profit"])),
            "corr_expected_profit_vs_profit": float(df["expected_profit_per_unit"].corr(df["profit"])),
            "corr_market_distance_vs_profit": float(df["market_distance"].corr(df["profit"])),
            "corr_edge_vs_profit": float(df["edge"].corr(df["profit"]))
        }

    def plot_ev_realization(self, save_path: Optional[Union[str, Path]] = None, show: bool = True) -> None:
        """Plots a 2x2 grid of EV realization diagnostics."""
        ev_curve = self.ev_vs_realized_roi()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. EV vs ROI
        ax = axes[0, 0]
        ax.plot(ev_curve["mean_ev_multiplier"], ev_curve["realized_roi_pct"], "o-", color="#2563eb", lw=2)
        ax.axhline(0, color="#666", ls="--")
        ax.set_xlabel("Mean EV Multiplier")
        ax.set_ylabel("Realized ROI (%)")
        ax.set_title("EV Bin: Expected vs Realized ROI")
        ax.grid(True, alpha=0.3)
        
        # 2. Expected Profit vs Realized Profit
        ax = axes[0, 1]
        ax.plot(ev_curve["mean_expected_profit"], ev_curve["realized_roi_pct"] / 100.0, "o-", color="#dc2626", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="#666", label="Perfect Realization")
        ax.set_xlabel("Mean Expected Profit (per unit)")
        ax.set_ylabel("Mean Realized Profit (per unit)")
        ax.set_title("EV Calibration: Expected vs Realized")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. False Positive Rate by Bin
        ax = axes[1, 0]
        ax.bar(ev_curve["ev_bin"], ev_curve["false_positive_rate_pct"], color="#f59e0b")
        ax.set_xlabel("EV Decile")
        ax.set_ylabel("Loss Rate (%)")
        ax.set_title("False +EV Rate by EV Bin")
        ax.grid(True, axis="y", alpha=0.3)
        
        # 4. Win Rate vs EV
        ax = axes[1, 1]
        ax.plot(ev_curve["mean_ev_multiplier"], ev_curve["win_rate_pct"], "o-", color="#16a34a", lw=2)
        ax.set_xlabel("Mean EV Multiplier")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate vs EV Bin")
        ax.grid(True, alpha=0.3)
        
        fig.suptitle("EV Realization Decomposition Audit", y=1.01, fontsize=14)
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def summary(self) -> str:
        """Generates a text-based diagnostic summary."""
        ev_curve = self.ev_vs_realized_roi()
        thresholds = self.threshold_sensitivity()
        corrs = self.compute_correlations()
        
        lines = [
            "=" * 60,
            "EV REALIZATION DECOMPOSITION REPORT",
            "=" * 60,
            f"Total events audited: {len(self.y_true)}",
            "",
            "1. CORRELATIONS (Expected vs Realized)",
            f"   EV Multiplier vs Profit:  {corrs['corr_ev_multiplier_vs_profit']:+.3f}",
            f"   Expected Profit vs Profit:{corrs['corr_expected_profit_vs_profit']:+.3f}",
            f"   Market Distance vs Profit:{corrs['corr_market_distance_vs_profit']:+.3f}",
            "",
            "2. EV DECILE BREAKDOWN",
        ]
        
        if len(ev_curve) > 1:
            low_ev = ev_curve.iloc[0]
            high_ev = ev_curve.iloc[-1]
            lines.append(f"   Lowest EV bin ROI:  {low_ev['realized_roi_pct']:+.2f}% (Mean EV: {low_ev['mean_ev_multiplier']:.3f})")
            lines.append(f"   Highest EV bin ROI: {high_ev['realized_roi_pct']:+.2f}% (Mean EV: {high_ev['mean_ev_multiplier']:.3f})")
            
            if high_ev["realized_roi_pct"] < low_ev["realized_roi_pct"]:
                lines.append("   ⚠️ WARNING: Inversion detected. Higher EV bins realize LOWER ROI.")
                lines.append("   ➡️ DIAGNOSIS: The model's 'edge' is likely driven by overconfidence,")
                lines.append("      not genuine predictive signal.")
        else:
            lines.append("   Not enough variance in EV to compute deciles.")
            
        lines.extend([
            "",
            "3. THRESHOLD SENSITIVITY",
        ])
        for _, row in thresholds.iterrows():
            if row["n_events"] > 0:
                lines.append(
                    f"   EV > {row['ev_threshold']:.2f}: "
                    f"Events={int(row['n_events'])} | "
                    f"ROI={row['realized_roi_pct']:+.2f}% | "
                    f"False Positives={row['false_positive_rate_pct']:.1f}%"
                )
                
        lines.extend(["", "=" * 60])
        return "\n".join(lines)