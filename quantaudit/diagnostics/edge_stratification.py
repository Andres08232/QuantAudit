"""
quantaudit.diagnostics.edge_stratification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diagnostic tool to evaluate the monetization of model edge in asymmetric 
payoff environments.

This module answers the question: "My model finds an edge against the market, 
but is that edge translating into positive ROI?" 

It stratifies performance by the model's edge, by the market's implied 
probability (to detect Favorite-Longshot Bias), and analyzes the divergence 
between expected edge and realized profit (Bucket Mismatch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, Optional, List

ArrayLike = Union[np.ndarray, pd.Series, list]

def _profit_factor(pnl: pd.Series) -> float:
    """Calculates the ratio of gross wins to gross losses."""
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss

class EdgeStratificationAuditor:
    """
    Audits the realization of model edge against market pricing.
    
    Attributes:
        y_true: Array of actual binary outcomes (1 = success, 0 = failure).
        model_prob: Array of predicted probabilities from the model.
        market_price: Array of market prices (e.g., decimal odds).
        closing_price: Optional array of closing market prices (for CLV analysis).
        stake: Fixed monetary stake per event.
    """

    def __init__(
        self,
        y_true: ArrayLike,
        model_prob: ArrayLike,
        market_price: ArrayLike,
        closing_price: Optional[ArrayLike] = None,
        stake: float = 1.0
    ):
        self.y_true = np.asarray(y_true, dtype=int)
        self.model_prob = np.asarray(model_prob, dtype=float)
        self.market_price = np.asarray(market_price, dtype=float)
        self.stake = float(stake)
        
        if closing_price is not None:
            self.closing_price = np.asarray(closing_price, dtype=float)
        else:
            self.closing_price = None

        self._validate_inputs()
        self._df: Optional[pd.DataFrame] = None

    def _validate_inputs(self):
        """Validates shapes and value ranges."""
        n = len(self.y_true)
        if not (len(self.model_prob) == len(self.market_price) == n):
            raise ValueError("y_true, model_prob, and market_price must have the same length.")
        
        if self.closing_price is not None and len(self.closing_price) != n:
            raise ValueError("closing_price must have the same length as y_true.")

        if not np.all((self.model_prob >= 0) & (self.model_prob <= 1)):
            raise ValueError("model_prob must be between 0 and 1.")
        if not np.all(self.market_price > 1.0):
            raise ValueError("market_price must be > 1.0 (valid decimal odds).")
        if not np.all(np.isin(self.y_true, [0, 1])):
            raise ValueError("y_true must contain only 0s and 1s.")

    def _build_base_dataframe(self) -> pd.DataFrame:
        """Calculates core financial and probabilistic metrics."""
        if self._df is not None:
            return self._df

        df = pd.DataFrame({
            "y_true": self.y_true,
            "model_prob": self.model_prob,
            "market_prob": 1.0 / self.market_price,
            "market_price": self.market_price
        })

        # Core Probabilistic Metrics
        df["edge"] = df["model_prob"] - df["market_prob"]
        df["ev_multiplier"] = df["model_prob"] * df["market_price"]
        
        # Financial Realization
        df["won"] = df["y_true"] == 1
        df["profit"] = np.where(
            df["won"], 
            self.stake * (df["market_price"] - 1.0), 
            -self.stake
        )
        
        # CLV (Closing Line Value) - Only if real closing prices are provided
        if self.closing_price is not None:
            df["closing_price"] = self.closing_price
            df["closing_prob"] = 1.0 / df["closing_price"]
            # CLV is positive if we bet at odds higher than the closing odds
            df["clv_pct"] = ((df["market_price"] / df["closing_price"]) - 1.0) * 100.0

        self._df = df
        return self._df

    def _bin_metrics(self, group: pd.DataFrame) -> Dict[str, float]:
        """Helper to calculate standard metrics for a bin."""
        n = len(group)
        if n == 0:
            return {"n_events": 0, "win_rate_pct": 0.0, "roi_pct": 0.0, "profit_factor": 0.0}
            
        wins = int(group["won"].sum())
        profit = float(group["profit"].sum())
        
        return {
            "n_events": n,
            "win_rate_pct": (wins / n) * 100.0,
            "roi_pct": (profit / (n * self.stake)) * 100.0,
            "profit_factor": _profit_factor(group["profit"]),
            "mean_model_prob": float(group["model_prob"].mean()),
            "mean_market_prob": float(group["market_prob"].mean()),
            "mean_edge_pct": float(group["edge"].mean() * 100.0),
            "mean_ev_multiplier": float(group["ev_multiplier"].mean())
        }

    def stratify_by_edge(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Stratifies events by model edge (Model Prob - Market Prob).
        
        Args:
            n_bins: Number of quantile bins to create.
            
        Returns:
            DataFrame with performance metrics per edge bin.
        """
        df = self._build_base_dataframe()
        
        try:
            df["edge_bin"] = pd.qcut(df["edge"], q=n_bins, duplicates="drop", labels=False)
        except ValueError:
            df["edge_bin"] = pd.cut(df["edge"], bins=n_bins, labels=False)

        rows = []
        for b, g in df.groupby("edge_bin", observed=True):
            metrics = self._bin_metrics(g)
            rows.append({"edge_bin": int(b), **metrics})
            
        return pd.DataFrame(rows)

    def stratify_by_implied_prob(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Stratifies events by the market's implied probability.
        Crucial for detecting Favorite-Longshot Bias (FLB).
        
        Args:
            n_bins: Number of bins for market probability.
            
        Returns:
            DataFrame with performance metrics per market probability bin.
        """
        df = self._build_base_dataframe()
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        df["prob_bin"] = pd.cut(df["market_prob"], bins=bins, include_lowest=True)

        rows = []
        for interval, g in df.groupby("prob_bin", observed=True):
            metrics = self._bin_metrics(g)
            metrics["prob_interval"] = f"{interval.left:.2f}-{interval.right:.2f}"
            rows.append(metrics)
            
        return pd.DataFrame(rows)

    def clv_analysis(self) -> Optional[pd.DataFrame]:
        """
        Analyzes Closing Line Value (CLV). Only works if closing_price was provided.
        Measures if the model is capturing value before the market adjusts.
        
        Returns:
            DataFrame with CLV statistics, or None if no closing prices.
        """
        df = self._build_base_dataframe()
        if "clv_pct" not in df.columns:
            return None

        # Stratify by CLV deciles
        try:
            df["clv_bin"] = pd.qcut(df["clv_pct"], q=10, duplicates="drop", labels=False)
        except ValueError:
            return pd.DataFrame() # Not enough variance in CLV

        rows = []
        for b, g in df.groupby("clv_bin", observed=True):
            metrics = self._bin_metrics(g)
            metrics["clv_bin"] = int(b)
            metrics["mean_clv_pct"] = float(g["clv_pct"].mean())
            rows.append(metrics)
            
        return pd.DataFrame(rows)

    def bucket_mismatch_analysis(self, ev_threshold: float = 1.05) -> pd.DataFrame:
        """
        Analyzes the divergence between expected edge and realized profit 
        for flagged "value" events (EV > threshold).
        
        A positive mismatch score means the model ranked the event high in edge, 
        but it realized low profit (or a loss).
        
        Args:
            ev_threshold: Minimum EV multiplier to consider an event a "value bet".
            
        Returns:
            DataFrame with mismatch statistics per edge decile.
        """
        df = self._build_base_dataframe()
        bets = df[df["ev_multiplier"] > ev_threshold].copy()
        
        if bets.empty:
            return pd.DataFrame()

        # Rank by edge and by realized profit
        bets["edge_rank"] = bets["edge"].rank(pct=True)
        bets["profit_rank"] = bets["profit"].rank(pct=True)
        
        # Bucket mismatch: Edge Rank - Profit Rank
        bets["mismatch_score"] = bets["edge_rank"] - bets["profit_rank"]
        
        try:
            bets["edge_bin"] = pd.qcut(bets["edge"], q=5, duplicates="drop", labels=False)
        except ValueError:
            return pd.DataFrame()

        rows = []
        for b, g in bets.groupby("edge_bin", observed=True):
            rows.append({
                "edge_bin": int(b),
                "n_bets": len(g),
                "mean_edge_pct": float(g["edge"].mean() * 100.0),
                "roi_pct": float(g["profit"].sum() / (len(g) * self.stake) * 100.0),
                "mean_mismatch_score": float(g["mismatch_score"].mean()),
                "corr_edge_profit": float(g["edge"].corr(g["profit"])) if len(g) > 2 else np.nan
            })
            
        return pd.DataFrame(rows)

    def plot_edge_stratification(self, save_path: Optional[Union[str, Path]] = None, show: bool = True) -> None:
        """Plots ROI and Win Rate by Edge Decile."""
        edge_data = self.stratify_by_edge()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color_roi = '#2563eb'
        color_wr = '#16a34a'
        
        ax1.set_xlabel('Edge Decile (0=Lowest Edge, 9=Highest Edge)')
        ax1.set_ylabel('ROI (%)', color=color_roi)
        ax1.bar(edge_data["edge_bin"], edge_data["roi_pct"], color=color_roi, alpha=0.6, label='ROI')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.tick_params(axis='y', labelcolor=color_roi)
        ax1.grid(True, axis='y', alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Win Rate (%)', color=color_wr)
        ax2.plot(edge_data["edge_bin"], edge_data["win_rate_pct"], color=color_wr, marker='o', linewidth=2, label='Win Rate')
        ax2.tick_params(axis='y', labelcolor=color_wr)
        
        plt.title('Edge Stratification: ROI vs Win Rate by Decile')
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def summary(self) -> str:
        """Generates an objective text summary of the edge monetization."""
        edge_data = self.stratify_by_edge()
        mismatch = self.bucket_mismatch_analysis()
        
        lines = [
            "=" * 60,
            "EDGE STRATIFICATION & MONETIZATION REPORT",
            "=" * 60,
            f"Total events audited: {len(self.y_true)}",
            f"Closing Line Value (CLV) data available: {'Yes' if self.closing_price is not None else 'No'}",
            "",
            "1. EDGE DECILE PERFORMANCE",
        ]
        
        if not edge_data.empty:
            top = edge_data.iloc[-1]
            bottom = edge_data.iloc[0]
            lines.append(f"   Lowest Edge Decile ROI:  {bottom['roi_pct']:+.2f}% (Mean Edge: {bottom['mean_edge_pct']:+.1f}pp)")
            lines.append(f"   Highest Edge Decile ROI: {top['roi_pct']:+.2f}% (Mean Edge: {top['mean_edge_pct']:+.1f}pp)")
            
            if top["roi_pct"] < bottom["roi_pct"]:
                lines.append("   ⚠️ ANOMALY: Edge Inversion detected. Higher edge deciles yield LOWER ROI.")
                lines.append("   ➡️ DIAGNOSIS: The model's edge is likely driven by overconfidence (miscalibration),")
                lines.append("      not genuine predictive signal. The market is pricing the risk correctly.")
            elif top["roi_pct"] > 0:
                lines.append("   ✅ POSITIVE: Highest edge deciles are monetizing positively.")
            else:
                lines.append("   ⚠️ NEGATIVE: Even the highest edge deciles are unprofitable.")
                
        lines.extend([
            "",
            "2. BUCKET MISMATCH (Value Bets Analysis)",
        ])
        
        if not mismatch.empty:
            avg_mismatch = mismatch["mean_mismatch_score"].mean()
            lines.append(f"   Average Mismatch Score: {avg_mismatch:+.3f}")
            if avg_mismatch > 0.1:
                lines.append("   ➡️ DIAGNOSIS: High positive mismatch. The model ranks events high in 'edge',")
                lines.append("      but they realize poor profits. The edge metric is anti-predictive for ROI.")
            else:
                lines.append("   ➡️ DIAGNOSIS: Mismatch is low. Edge ranking aligns reasonably with realized profits.")
        else:
            lines.append("   Not enough value bets (EV > threshold) to compute mismatch.")
            
        lines.extend(["", "=" * 60])
        return "\n".join(lines)