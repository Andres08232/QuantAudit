# QuantAudit

**QuantAudit** is an open-source Python library designed to audit, evaluate, and diagnose probabilistic models. 

Originally born from the ashes of *VisionGoat* (a sports betting predictive engine that proved the market is too efficient to beat with public data), QuantAudit repurposes its rigorous quantitative auditing framework for general use.

## Features
- **Calibration Audits:** Log Loss, Brier Score, ECE, OCI, and Tail Error.
- **EV Decomposition:** Diagnose if your model is finding signal or just measuring its own biases.
- **Edge Stratification:** Identify if model confidence is inversely proportional to profitability.
- **ROI Attribution:** Attribute PnL to calibration, edge, variance, and market margin.

## Installation
```bash
pip install -e .
(Más detalles de uso en la carpeta examples/)