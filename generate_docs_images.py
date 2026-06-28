import matplotlib
matplotlib.use('Agg') # Backend no interactivo para evitar imágenes en blanco

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'

output_dir = Path("docs/images")
output_dir.mkdir(parents=True, exist_ok=True)

print(" Generando imágenes con datos reales...")

# 1. Generar datos reales y entrenar modelo
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, 
                           n_redundant=5, random_state=42, weights=[0.4, 0.6])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest suele estar descalibrado (overconfident), ideal para QuantAudit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
model_probs = model.predict_proba(X_test)[:, 1]

# Simular "market_price" (cuotas implícitas) para los módulos financieros
# Asumimos que el mercado es ligeramente más eficiente que el modelo
market_probs = np.clip(model_probs + np.random.normal(0, 0.05, len(model_probs)), 0.05, 0.95)
market_price = 1.0 / market_probs

# ==========================================
# 1. Calibration Curve (Reliability Diagram)
# ==========================================
print("Generando calibration_curve.png...")
fig, ax = plt.subplots(figsize=(8, 6))
bins = np.linspace(0.0, 1.0, 11)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

bin_true = []
bin_pred = []
for i in range(len(bins)-1):
    mask = (model_probs >= bins[i]) & (model_probs < bins[i+1])
    if mask.sum() > 0:
        bin_true.append(y_test[mask].mean())
        bin_pred.append(model_probs[mask].mean())

ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', alpha=0.7, linewidth=2)
ax.plot(bin_pred, bin_true, 's-', color='#e74c3c', label='RandomForest (Uncalibrated)', linewidth=2, markersize=8)

ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax.set_title('Reliability Curve: Calibration Audit', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_dir / "calibration_curve.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print("✅ calibration_curve.png guardada")

# ==========================================
# 2. EV Decomposition (Theoretical vs Realized)
# ==========================================
print("Generando ev_decomposition.png...")
fig, ax = plt.subplots(figsize=(8, 6))
n_events = len(y_test)
# Calcular EV teórico y ROI real acumulado
ev_theoretical = np.cumsum((model_probs * market_price) - 1)
roi_realized = np.cumsum((y_test * market_price) - 1)

ax.plot(range(1, n_events + 1), ev_theoretical, label='Theoretical EV (Cumulative)', color='#2ecc71', linewidth=2)
ax.plot(range(1, n_events + 1), roi_realized, label='Realized ROI (Cumulative)', color='#e74c3c', linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

ax.set_xlabel('Number of Events', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax.set_title('EV Decomposition: Theoretical Edge vs Realized ROI', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / "ev_decomposition.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print("✅ ev_decomposition.png guardada")

# ==========================================
# 3. Edge Stratification (ROI by Decile)
# ==========================================
print("Generando edge_stratification.png...")
fig, ax = plt.subplots(figsize=(8, 6))
deciles = np.arange(1, 11)
roi_by_decile = []

# Calcular ROI por decil de edge (diferencia entre prob modelo y prob mercado)
edge = model_probs - market_probs
for i in range(10):
    lower = np.percentile(edge, i * 10)
    upper = np.percentile(edge, (i + 1) * 10)
    mask = (edge >= lower) & (edge <= upper)
    if mask.sum() > 0:
        roi = ((y_test[mask] * market_price[mask]) - 1).mean() * 100
        roi_by_decile.append(roi)
    else:
        roi_by_decile.append(0)

colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in roi_by_decile]
bars = ax.bar(deciles, roi_by_decile, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax.set_xlabel('Edge Decile (1=Lowest Edge, 10=Highest Edge)', fontsize=12, fontweight='bold')
ax.set_ylabel('Realized ROI (%)', fontsize=12, fontweight='bold')
ax.set_title('Edge Stratification: Favorite-Longshot Bias Detection', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(deciles)

plt.tight_layout()
plt.savefig(output_dir / "edge_stratification.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print("✅ edge_stratification.png guardada")

# ==========================================
# 4. ROI Attribution (Performance vs Threshold)
# ==========================================
print("Generando roi_attribution.png...")
fig, ax = plt.subplots(figsize=(8, 6))
thresholds = [1.00, 1.05, 1.10, 1.15, 1.20]
roi_model_a = []
roi_model_b = []

# Simular dos modelos: el original y uno ligeramente ajustado
model_b_probs = np.clip(model_probs * 1.05, 0, 1)

for thresh in thresholds:
    # Modelo A
    mask_a = (model_probs * market_price) >= thresh
    if mask_a.sum() > 0:
        roi_a = ((y_test[mask_a] * market_price[mask_a]) - 1).mean() * 100
        roi_model_a.append(roi_a)
    else:
        roi_model_a.append(0)
        
    # Modelo B
    mask_b = (model_b_probs * market_price) >= thresh
    if mask_b.sum() > 0:
        roi_b = ((y_test[mask_b] * market_price[mask_b]) - 1).mean() * 100
        roi_model_b.append(roi_b)
    else:
        roi_model_b.append(0)

ax.plot(thresholds, roi_model_a, marker='o', label='Model A (Naive)', color='#e74c3c', linewidth=2, markersize=8)
ax.plot(thresholds, roi_model_b, marker='s', label='Model B (Adjusted)', color='#2ecc71', linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

ax.set_xlabel('EV Threshold (Minimum required edge)', fontsize=12, fontweight='bold')
ax.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
ax.set_title('ROI Attribution: Model Performance vs Strictness', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='best')

plt.tight_layout()
plt.savefig(output_dir / "roi_attribution.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print("✅ roi_attribution.png guardada")

print("\n🎉 ¡Todas las imágenes reales se generaron correctamente en docs/images/!")