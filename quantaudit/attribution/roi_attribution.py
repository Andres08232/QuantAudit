"""
quantaudit.attribution.roi_attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threshold-based backtesting and economic attribution for probabilistic models 
in asymmetric payoff environments.

This module does NOT find profitable betting strategies. It answers the question:
"If I filter my model's predictions by a minimum Expected Value (EV) threshold, 
how does the historical ROI, Profit Factor, and Max Drawdown behave for 
multiple competing models?"

It is designed to compare the economic realization of different probabilistic 
models across varying levels of strictness (EV thresholds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, Optional, List

ArrayLike = Union[np.ndarray, pd.Series, list]

def _max_drawdown(pnl_series: pd.Series) -> float:
    """Calculates the maximum peak-to-trough drawdown of a PnL series."""
    if pnl_series.empty:
        return 0.0
    cumulative = pnl_series.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    return float(drawdown.max())

def _profit_factor(pnl_series: pd.Series) -> float:
    """Calculates the ratio of gross wins to gross losses."""
    gross_win = float(pnl_series[pnl_series > 0].sum())
    gross_loss = float(abs(pnl_series[pnl_series < 0].sum()))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss

class ROIAttributionAuditor:
    """
    Audits the economic performance of multiple probabilistic models across 
    different Expected Value (EV) selection thresholds.
    
    Attributes:
        y_true: Array of actual binary outcomes (1 = success, 0 = failure).
        models: Dictionary mapping model names to their predicted probabilities.
        market_price: Array of market prices (e.g., decimal odds).
        thresholds: List of EV multiplier cutoffs to test.
        stake: Fixed monetary stake per event.
    """

    def __init__(
        self,
        y_true: ArrayLike,
        models: Dict[str, ArrayLike],
        market_price: ArrayLike,
        thresholds: List[float] = None,
        stake: float = 1.0
    ):
        self.y_true = np.asarray(y_true, dtype=int)
        self.models = {name: np.asarray(probs, dtype=float) for name, probs in models.items()}
        self.market_price = np.asarray(market_price, dtype=float)
        self.thresholds = thresholds if thresholds is not None else [1.00, 1.02, 1.05, 1.10]
        self.stake = float(stake)
        
        self._validate_inputs()
        self._market_prob = 1.0 / self.market_price

    def _validate_inputs(self):
        """Validates shapes and value ranges."""
        n = len(self.y_true)
        if not np.all(np.isin(self.y_true, [0, 1])):
            raise ValueError("y_true must contain only 0s and 1s.")
        if not np.all(self.market_price > 1.0):
            raise ValueError("market_price must be > 1.0 (valid decimal odds).")
            
        for name, probs in self.models.items():
            if len(probs) != n:
                raise ValueError(f"Model '{name}' probabilities length mismatch.")
            if not np.all((probs >= 0) & (probs <= 1)):
                raise ValueError(f"Model '{name}' probabilities must be between 0 and 1.")

    def run_backtest(self) -> pd.DataFrame:
        """
        Executes a flat-stake backtest for every model across every EV threshold.
        
        Note: The input arrays are assumed to be in chronological order for 
        accurate Max Drawdown calculation.
        
        Returns:
            A pandas DataFrame containing the financial metrics for each 
            model-threshold combination.
        """
        rows = []
        
        for model_name, probs in self.models.items():
            edge = probs - self._market_prob
            ev_multiplier = probs * self.market_price
            
            won = self.y_true == 1
            pnl_all = np.where(won, self.stake * (self.market_price - 1.0), -self.stake)
            
            for thr in self.thresholds:
                mask = ev_multiplier > thr
                n_bets = int(mask.sum())
                
                if n_bets == 0:
                    rows.append({
                        "model": model_name, "ev_threshold": thr, "n_bets": 0,
                        "win_rate_pct": 0.0, "total_profit": 0.0, "roi_pct": 0.0,
                        "profit_factor": 0.0, "max_drawdown": 0.0, 
                        "mean_edge_pct": 0.0, "mean_ev_multiplier": 0.0
                    })
                    continue
                
                pnl_filtered = pd.Series(pnl_all[mask])
                wins = int(self.y_true[mask].sum())
                profit = float(pnl_filtered.sum())
                
                rows.append({
                    "model": model_name,
                    "ev_threshold": thr,
                    "n_bets": n_bets,
                    "win_rate_pct": (wins / n_bets) * 100.0,
                    "total_profit": profit,
                    "roi_pct": (profit / (n_bets * self.stake)) * 100.0,
                    "profit_factor": _profit_factor(pnl_filtered),
                    "max_drawdown": _max_drawdown(pnl_filtered),
                    "mean_edge_pct": float(edge[mask].mean() * 100.0),
                    "mean_ev_multiplier": float(ev_multiplier[mask].mean())
                })
                
        return pd.DataFrame(rows)

    def get_edge_distribution(self) -> pd.DataFrame:
        """
        Calculates the statistical distribution of the model's edge and EV 
        multiplier against the market.
        
        Returns:
            A pandas DataFrame with distribution metrics per model.
        """
        rows = []
        for model_name, probs in self.models.items():
            edge = probs - self._market_prob
            ev = probs * self.market_price
            
            rows.append({
                "model": model_name,
                "n_events": len(probs),
                "edge_mean_pct": float(edge.mean() * 100.0),
                "edge_median_pct": float(np.median(edge) * 100.0),
                "edge_p90_pct": float(np.percentile(edge, 90) * 100.0),
                "ev_mean": float(ev.mean()),
                "ev_p90": float(np.percentile(ev, 90)),
                "freq_ev_gt_105_pct": float((ev > 1.05).sum() / len(ev) * 100.0)
            })
        return pd.DataFrame(rows)

    def plot_roi_vs_threshold(self, save_path: Optional[Union[str, Path]] = None, show: bool = True) -> None:
        """
        Plots the ROI curve across EV thresholds for all models.
        
        Args:
            save_path: Optional path to save the figure.
            show: Whether to display the plot.
        """
        backtest_df = self.run_backtest()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in self.models.keys():
            subset = backtest_df[backtest_df["model"] == model_name]
            ax.plot(subset["ev_threshold"], subset["roi_pct"], marker="o", linewidth=2, label=model_name)
            
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("EV Threshold (model_prob * market_price)")
        ax.set_ylabel("Realized ROI (%)")
        ax.set_title("ROI Attribution: Flat-Stake ROI vs EV Selection Threshold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def summary(self) -> str:
        """Generates an objective text summary of the threshold backtest."""
        backtest_df = self.run_backtest()
        dist_df = self.get_edge_distribution()
        
        lines = [
            "=" * 70,
            "THRESHOLD BACKTEST & ROI ATTRIBUTION REPORT",
            "=" * 70,
            f"Total events evaluated: {len(self.y_true)}",
            f"Models compared: {', '.join(self.models.keys())}",
            f"Thresholds tested: {self.thresholds}",
            "",
            "1. EDGE DISTRIBUTION (Model vs Market)",
        ]
        
        for _, row in dist_df.iterrows():
            lines.append(
                f"   {row['model']}: Mean Edge={row['edge_mean_pct']:+.2f}pp | "
                f"Mean EV={row['ev_mean']:.3f} | "
                f"% events with EV>1.05: {row['freq_ev_gt_105_pct']:.1f}%"
            )
            
        lines.extend(["", "2. BACKTEST PERFORMANCE BY THRESHOLD"])
        
        for thr in self.thresholds:
            lines.append(f"   --- EV > {thr:.2f} ---")
            subset = backtest_df[backtest_df["ev_threshold"] == thr]
            for _, row in subset.iterrows():
                if row["n_bets"] > 0:
                    lines.append(
                        f"   {row['model']}: Bets={int(row['n_bets'])} | "
                        f"ROI={row['roi_pct']:+.2f}% | "
                        f"PF={row['profit_factor']:.2f} | "
                        f"MaxDD={row['max_drawdown']:.2f}"
                    )
                else:
                    lines.append(f"   {row['model']}: No bets qualified.")
                    
        lines.extend(["", "=" * 70])
        return "\n".join(lines)