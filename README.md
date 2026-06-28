# QuantAudit

**QuantAudit** is an open-source Python library designed to audit, evaluate, and diagnose probabilistic models in asymmetric payoff environments.

Originally born from the ashes of **VisionGoat** (a sports betting predictive engine that proved the market is too efficient to beat with public data), QuantAudit repurposes its rigorous quantitative auditing framework for general use. It is not a tool to "find winning bets"; it is a **forensic scalpel** to diagnose exactly *why* your model is destroying capital.

---

## 🚀 Why QuantAudit?

### Benchmarking vs. Market

Standard libraries like `scikit-learn` are great for general ML, and `pyfolio` is great for continuous financial returns. But when you need to make high-stakes decisions based on extreme probabilities in binary events (e.g., sports betting, algorithmic trading, credit risk, churn prediction), they fall short.

| Feature                                 | `scikit-learn` | `pyfolio` / `quantstats` | **QuantAudit** |
| :-------------------------------------- | :------------: | :----------------------: | :------------: |
| Log Loss / Brier Score                  |        ✅       |             ❌            |        ✅       |
| Expected Calibration Error (ECE)        |        ❌       |             ❌            |        ✅       |
| **Overconfidence Index (OCI)**          |        ❌       |             ❌            |        ✅       |
| **Tail Error (Top X% Confidence)**      |        ❌       |             ❌            |        ✅       |
| **EV Realization & Bucket Mismatch**    |        ❌       |             ❌            |        ✅       |
| **Edge Stratification & FLB Detection** |        ❌       |             ❌            |        ✅       |
| Multi-Model Threshold Backtesting       |   ❌ (Manual)   |             ❌            |   ✅ (Native)   |

### The QuantAudit Difference

We don't just tell you if your model is calibrated; we tell you:

* **Where** it is overconfident.
* **Whether** your edge is an illusion.
* **Why** your EV selection is failing to monetize.

---

# The 4-Step Autopsy Pipeline

QuantAudit structures the diagnostic process into four sequential modules.

## 1. `calibration`

**Question:** *Is the model lying about probabilities?*

Evaluates global and tail calibration:

* Log Loss
* Brier Score
* Expected Calibration Error (ECE)
* Overconfidence Index (OCI)
* Tail Error

---

## 2. `diagnostics.ev_decomposition`

**Question:** *Is the Expected Value formula broken?*

Decomposes Expected Value into realized outcomes to detect whether the model is finding genuine signal or merely measuring its own biases.

---

## 3. `diagnostics.edge_stratification`

**Question:** *Is the theoretical edge translating into real money?*

Stratifies ROI by:

* Edge deciles
* Market implied probability

Detects:

* Favorite-Longshot Bias (FLB)
* Edge Inversion

---

## 4. `attribution.roi_attribution`

**Question:** *How do models perform under increasing strictness?*

Executes flat-stake backtests across multiple models and EV thresholds while calculating:

* ROI
* Profit Factor
* Maximum Drawdown

---

# Installation

Clone the repository and install it in editable mode.

```bash
git clone https://github.com/TU_USUARIO/quantaudit.git
cd quantaudit
pip install -e .
```

---

# Quick Start

```python
import numpy as np
from quantaudit import (
    CalibrationAuditor,
    EVDecompositionAuditor,
    EdgeStratificationAuditor,
    ROIAttributionAuditor
)

# Generate synthetic data (or load your own)
np.random.seed(42)

n = 1000
y_true = np.random.binomial(1, 0.4, n)

market_price = 1.0 / np.random.uniform(0.3, 0.7, n)

model_probs = np.clip(
    y_true * 0.5 + np.random.normal(0.25, 0.2, n),
    0.01,
    0.99
)

# Calibration Audit
cal_audit = CalibrationAuditor(
    y_true,
    {"MyModel": model_probs}
)

print(cal_audit.summary())
cal_audit.plot_reliability_curves()

# EV Decomposition
ev_audit = EVDecompositionAuditor(
    y_true,
    model_probs,
    market_price=market_price
)

print(ev_audit.summary())

# Edge Stratification
edge_audit = EdgeStratificationAuditor(
    y_true,
    model_probs,
    market_price=market_price
)

print(edge_audit.summary())

# ROI Attribution
roi_audit = ROIAttributionAuditor(
    y_true,
    models={
        "Model_A": model_probs,
        "Model_B": model_probs * 1.05,
    },
    market_price=market_price,
    thresholds=[1.00, 1.05, 1.10]
)

roi_audit.plot_roi_vs_threshold()
```

---

# Testing

QuantAudit includes unit tests to ensure the mathematical integrity of the financial and probabilistic metrics.

Run the test suite with:

```bash
pip install pytest
pytest tests/
```

---

# Roadmap

* [x] Calibration Audit (Log Loss, Brier, ECE, OCI, Tail Error)
* [x] EV Realization Decomposition (Signal vs. Bias)
* [x] Edge Stratification & Favorite-Longshot Bias Detection
* [x] ROI Attribution & Threshold Backtesting
* [ ] Auto-Recalibration (Temperature Scaling, Isotonic Regression)
* [ ] Multi-class probabilistic audit (e.g., 1X2 markets simultaneously)
* [ ] Kelly Criterion stake optimization integration
