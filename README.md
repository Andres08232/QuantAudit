# QuantAudit

**QuantAudit** is an open-source Python library designed to audit, evaluate, and diagnose probabilistic models. 

Originally born from the ashes of *VisionGoat* (a sports betting predictive engine that proved the market is too efficient to beat with public data), QuantAudit repurposes its rigorous quantitative auditing framework for general use.

##  Why QuantAudit? (Benchmarking vs Market)

Standard libraries like `scikit-learn` are great for general ML, but they fall short when you need to make high-stakes decisions based on extreme probabilities (e.g., sports betting, algorithmic trading, credit risk).

| Feature | `scikit-learn` | `torchmetrics` | **QuantAudit** |
| :--- | :---: | :---: | :---: |
| Log Loss / Brier Score | ✅ | ✅ | ✅ |
| Expected Calibration Error (ECE) | ❌ | ✅ | ✅ |
| **Overconfidence Index (OCI)** | ❌ | ❌ | ✅ |
| **Tail Error (Top X% Confidence)** | ❌ | ❌ | ✅ |
| Multi-Model Comparison & Plotting | ❌ (Manual) | ❌ | ✅ (Native) |
| **Business/ROI Diagnostics** | ❌ | ❌ | ✅ (Upcoming) |

**The QuantAudit Difference:** We don't just tell you if your model is calibrated; we tell you **where it is overconfident** and **why it might be destroying your capital**.

##  Installation

```bash
# Clone the repo
git clone https://github.com/tu-usuario/quantaudit.git
cd quantaudit

# Install in editable mode
pip install -e .

Quick Start
from quantaudit.calibration import CalibrationAuditor

# y_true: 0s and 1s. model_probs: dict of {"ModelName": array_of_probs}
auditor = CalibrationAuditor(y_true, model_probs)

# Get a pandas DataFrame with LogLoss, Brier, ECE, OCI, TailError
print(auditor.compute_metrics())

# Plot reliability curves for all models at once
auditor.plot_reliability_curves()
🗺️ Roadmap

    Calibration Audit (LogLoss, Brier, ECE, OCI, Tail Error)
    EV Realization Decomposition (Signal vs Bias)
    Edge Stratification & ROI Attribution
    Auto-Recalibration (Temperature Scaling, Isotonic)