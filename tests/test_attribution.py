"""
tests.test_attribution
~~~~~~~~~~~~~~~~~~~~~~

Basic unit tests for the ROI Attribution module.
"""
import pytest
import numpy as np
import pandas as pd
from quantaudit.attribution.roi_attribution import ROIAttributionAuditor

def test_perfect_model_roi():
    """A model that perfectly predicts outcomes should yield positive ROI."""
    y_true = np.array([1, 0, 1, 0])
    # Model predicts 0.9 for wins, 0.1 for losses
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    # Market odds are 2.0 for all (implied prob 0.5)
    market_price = np.array([2.0, 2.0, 2.0, 2.0])
    
    auditor = ROIAttributionAuditor(y_true, {"Perfect": probs}, market_price, thresholds=[1.0])
    df = auditor.run_backtest()
    
    # 2 wins at 2.0 odds (+2 profit each), 0 losses. Total profit = 4. Stake = 4. ROI = 100%
    assert df.iloc[0]["roi_pct"] == 100.0
    assert df.iloc[0]["profit_factor"] == float("inf")

def test_threshold_filtering():
    """Ensure that raising the threshold correctly filters out low EV bets."""
    y_true = np.array([1, 0])
    probs = np.array([0.6, 0.4]) # EV multipliers: 1.2, 0.8
    market_price = np.array([2.0, 2.0])
    
    auditor = ROIAttributionAuditor(
        y_true, {"Model": probs}, market_price, thresholds=[1.0, 1.1]
    )
    df = auditor.run_backtest()
    
    # At threshold 1.0, both bets qualify (EV > 1.0 and 0.8? Wait, 0.8 is not > 1.0)
    # Let's correct: EV multipliers are 1.2 and 0.8.
    # Threshold 1.0: only 1 bet qualifies (the 1.2 one).
    # Threshold 1.1: only 1 bet qualifies (the 1.2 one).
    assert df[df["ev_threshold"] == 1.0].iloc[0]["n_bets"] == 1
    assert df[df["ev_threshold"] == 1.1].iloc[0]["n_bets"] == 1

def test_invalid_inputs():
    """Ensure the auditor rejects invalid probabilities or odds."""
    y_true = np.array([1, 0])
    probs = np.array([0.5, 0.5])
    bad_odds = np.array([0.5, 0.5]) # Odds must be > 1.0
    
    with pytest.raises(ValueError, match="market_price must be > 1.0"):
        ROIAttributionAuditor(y_true, {"Model": probs}, bad_odds)