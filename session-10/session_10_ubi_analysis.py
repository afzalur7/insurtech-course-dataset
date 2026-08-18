# ============================================================================
# Session 10 — Usage-Based Insurance: UBI Data Analysis (COMPLETE RUNNABLE SCRIPT)
# InsurTech & Digital Risk Solutions (MBA) — Woxsen University
#
# HOW TO RUN:
#   venv/bin/python session_10_ubi_analysis.py
#   (or run top-to-bottom in one Jupyter notebook — the page's §5.1–§5.4 blocks
#    are merged here in order so nothing is left undefined)
#
# DATA FILE: data/telematics_data.csv (provided) — 500 drivers, one month of
#            synthetic driving behaviour. Falls back to telematics_data.csv
#            in the current folder.
#
# OUTPUTS:   console tables + ubi_premium_comparison.png + ubi_risk_scatter.png
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# SECTION 5.1 — Loading the Telematics Data
# ----------------------------------------------------------------------------
try:
    telematics = pd.read_csv('data/telematics_data.csv')
except FileNotFoundError:
    telematics = pd.read_csv('telematics_data.csv')

print(f"Drivers: {telematics.shape[0]:,}")
print(f"Columns: {telematics.columns.tolist()}")
print(f"\nData types:\n{telematics.dtypes}")
print(f"\nSummary statistics:\n{telematics.describe().round(1)}")

# Key columns:
# driver_id — Unique driver identifier
# total_km — Kilometres driven in the month
# avg_speed_kmph — Average speed
# hard_braking_events — Count of hard braking events
# hard_accel_events — Count of rapid acceleration events
# night_driving_pct — Percentage of driving at night (10pm-5am)
# phone_use_events — Count of phone usage events while driving
# previous_claims — Number of claims in the last 3 years
# claim_amount_last_3y — Total claim amount in the last 3 years

# ----------------------------------------------------------------------------
# SECTION 5.2 — Building a Behavioural Risk Score
# ----------------------------------------------------------------------------
RISK_WEIGHTS = {
    'km_risk': 0.20,              # Higher mileage = more exposure
    'braking_risk': 0.25,         # Hard braking = unsafe driving pattern
    'accel_risk': 0.15,           # Hard acceleration = aggressive driving
    'night_risk': 0.20,           # Night driving = higher accident risk
    'phone_risk': 0.20            # Phone use = distraction risk
}

KM_THRESHOLD_HIGH = 2000       # km/month — above this is high mileage
BRAKING_THRESHOLD = 15         # events/month — above this is aggressive
ACCEL_THRESHOLD = 10           # events/month
NIGHT_THRESHOLD = 25           # % of driving at night
PHONE_THRESHOLD = 5            # events/month

telematics['km_risk'] = np.clip((telematics['total_km'] / KM_THRESHOLD_HIGH) * 100, 0, 100)
telematics['braking_risk'] = np.clip((telematics['hard_braking_events'] / BRAKING_THRESHOLD) * 100, 0, 100)
telematics['accel_risk'] = np.clip((telematics['hard_accel_events'] / ACCEL_THRESHOLD) * 100, 0, 100)
telematics['night_risk'] = np.clip((telematics['night_driving_pct'] / NIGHT_THRESHOLD) * 100, 0, 100)
telematics['phone_risk'] = np.clip((telematics['phone_use_events'] / PHONE_THRESHOLD) * 100, 0, 100)

telematics['risk_score'] = (
    telematics['km_risk'] * RISK_WEIGHTS['km_risk'] +
    telematics['braking_risk'] * RISK_WEIGHTS['braking_risk'] +
    telematics['accel_risk'] * RISK_WEIGHTS['accel_risk'] +
    telematics['night_risk'] * RISK_WEIGHTS['night_risk'] +
    telematics['phone_risk'] * RISK_WEIGHTS['phone_risk']
)

telematics['risk_tier'] = pd.cut(
    telematics['risk_score'],
    bins=[0, 20, 40, 60, 100],
    labels=['Very Low', 'Low', 'Moderate', 'High'],
    include_lowest=True
)

print("Risk Score Distribution:")
print(telematics['risk_tier'].value_counts().sort_index())
print(f"\nMean risk score: {telematics['risk_score'].mean():.1f}")
print(f"Median risk score: {telematics['risk_score'].median():.1f}")
print(f"Std dev: {telematics['risk_score'].std():.1f}")

# ----------------------------------------------------------------------------
# SECTION 5.3 — PAYD vs. PHYD Pricing Comparison
# ----------------------------------------------------------------------------
PAYD_RATE = 1.50  # ₹1.50 per km
PHYD_BASE_RATE = 0.80        # ₹0.80 per km (lower base because behaviour-adjusted)
PHYD_RISK_MULTIPLIER = {
    'Very Low': 0.60,
    'Low': 0.85,
    'Moderate': 1.10,
    'High': 1.50
}

telematics['payd_premium'] = telematics['total_km'] * PAYD_RATE
# pandas 3.x: map a categorical with a dict can return a non-arithmetic dtype,
# so cast to str first, then to float.
telematics['phyd_rate'] = telematics['risk_tier'].astype(str).map(PHYD_RISK_MULTIPLIER).astype(float)
telematics['phyd_premium'] = telematics['total_km'] * PHYD_BASE_RATE * telematics['phyd_rate']

TRADITIONAL_FLAT = 2500
telematics['traditional_premium'] = TRADITIONAL_FLAT
telematics['payd_savings'] = TRADITIONAL_FLAT - telematics['payd_premium']
telematics['phyd_savings'] = TRADITIONAL_FLAT - telematics['phyd_premium']

comparison = telematics.groupby('risk_tier', observed=False).agg(
    driver_count=('driver_id', 'count'),
    avg_km=('total_km', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_payd=('payd_premium', 'mean'),
    avg_phyd=('phyd_premium', 'mean'),
    avg_traditional=('traditional_premium', 'mean'),
    avg_payd_savings=('payd_savings', 'mean'),
    avg_phyd_savings=('phyd_savings', 'mean'),
).round(0)

print("=" * 110)
print(f"{'Risk Tier':12s} {'Drivers':>8s} {'Avg Km':>8s} {'Risk':>6s} {'PAYD ₹':>9s} {'PHYD ₹':>9s} {'Trad ₹':>9s} {'PAYD Save':>10s} {'PHYD Save':>10s}")
print("=" * 110)
for tier in ['Very Low', 'Low', 'Moderate', 'High']:
    r = comparison.loc[tier]
    print(f"{tier:12s} {r['driver_count']:>5.0f}   {r['avg_km']:>5,.0f}  {r['avg_risk_score']:>4.0f}  ₹{r['avg_payd']:>6,.0f}  ₹{r['avg_phyd']:>6,.0f}  ₹{r['avg_traditional']:>5,.0f}  {'+' if r['avg_payd_savings'] > 0 else ''}₹{r['avg_payd_savings']:>6,.0f}  {'+' if r['avg_phyd_savings'] > 0 else ''}₹{r['avg_phyd_savings']:>6,.0f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

tiers = ['Very Low', 'Low', 'Moderate', 'High']
x = np.arange(len(tiers))
width = 0.25

ax1.bar(x - width, comparison.loc[tiers, 'avg_traditional'], width, label='Traditional (Flat)', color='#a29bfe', alpha=0.7)
ax1.bar(x, comparison.loc[tiers, 'avg_payd'], width, label='PAYD', color='#6c5ce7')
ax1.bar(x + width, comparison.loc[tiers, 'avg_phyd'], width, label='PHYD', color='#00d2d3')
ax1.set_xlabel('Risk Tier')
ax1.set_ylabel('Monthly Premium (₹)')
ax1.set_title('Premium Comparison by Risk Tier', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tiers)
ax1.legend()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

if 'claim_amount' in telematics.columns:
    ax2.scatter(telematics['risk_score'], telematics['claim_amount'],
               alpha=0.4, color='#6c5ce7', s=20)
    z = np.polyfit(telematics['risk_score'], telematics['claim_amount'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(telematics['risk_score'].min(), telematics['risk_score'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    corr = telematics['risk_score'].corr(telematics['claim_amount'])
    ax2.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95),
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_title('Risk Score vs. Claim Amount', fontweight='bold')
    ax2.set_xlabel('Behavioural Risk Score (0–100)')
    ax2.set_ylabel('Claim Amount (₹)')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('ubi_premium_comparison.png', dpi=120)   # artifact for submission
plt.show()

# ----------------------------------------------------------------------------
# SECTION 5.4 — Interpreting the Results (worked examples from the page)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 5.4 — Interpretation examples")
print("=" * 70)

aggressive_low_km = 500     # km/month, aggressive driving (High tier)
payd = aggressive_low_km * PAYD_RATE
phyd = aggressive_low_km * PHYD_BASE_RATE * 1.50
print(f"Aggressive low-mileage driver (500 km, High tier):")
print(f"  PAYD = ₹{payd:.0f}  |  PHYD = ₹{phyd:.0f}  |  Traditional = ₹{TRADITIONAL_FLAT}")
print(f"  -> PAYD underprices the risk; PHYD adjusts it up by the behavioural multiplier.")

safe_high_km = 3000        # km/month, safe driving (Very Low tier)
payd = safe_high_km * PAYD_RATE
phyd = safe_high_km * PHYD_BASE_RATE * 0.60
print(f"\nSafe high-mileage driver (3000 km, Very Low tier):")
print(f"  PAYD = ₹{payd:.0f}  |  PHYD = ₹{phyd:.0f}  |  Traditional = ₹{TRADITIONAL_FLAT}")
print(f"  -> PAYD overcharges the safe driver; PHYD rewards them with a discount.")

print("\nCharts saved: ubi_premium_comparison.png")
