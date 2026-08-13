# ============================================================================
# Session 09 — Digital Customer Acquisition (COMPLETE RUNNABLE SCRIPT)
# InsurTech & Digital Risk Solutions (MBA) — Woxsen University
#
# HOW TO RUN:
#   venv/bin/python session_09_digital_acquisition.py
#   (or run top-to-bottom in one Jupyter notebook — the page's §5.1–§6.2
#    blocks are merged here in order so nothing is left undefined)
#
# DATA FILE: data/funnel_data.csv (provided) — the script also looks for
#            funnel_data.csv in the current folder as a fallback.
#
# OUTPUTS:   console tables + chapter9_funnel.png + channel_economics_bubble.png
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# SECTION 5.1 — Loading and Preparing Funnel Data
# Each row represents a user session with 1/0 flags for each funnel stage.
# ----------------------------------------------------------------------------
try:
    funnel_data = pd.read_csv('data/funnel_data.csv')
except FileNotFoundError:
    funnel_data = pd.read_csv('funnel_data.csv')

print(f"Total sessions tracked: {len(funnel_data):,}")
print(f"\nColumns: {funnel_data.columns.tolist()}")
print(f"\nChannels: {funnel_data['channel'].value_counts().to_dict()}")

# Calculate stage-wise conversion
funnel_stages = ['stage_awareness', 'stage_consideration',
                 'stage_quote', 'stage_purchase', 'stage_onboarding']

stage_counts = {}
for stage in funnel_stages:
    stage_counts[stage] = funnel_data[stage].sum()

print("\nStage-wise user counts:")
for stage, count in stage_counts.items():
    print(f"  {stage:25s}: {count:6,.0f}")

# ----------------------------------------------------------------------------
# SECTION 5.2 — Calculating Conversion Rates
# ----------------------------------------------------------------------------
# Overall conversion (top-of-funnel to end)
overall_conv = stage_counts['stage_onboarding'] / stage_counts['stage_awareness']
print(f"Overall conversion (Awareness → Onboarding): {overall_conv*100:.1f}%")

# Stage-by-stage conversion
stages_list = list(funnel_stages)
conversion_rates = {}
for i in range(len(stages_list) - 1):
    from_stage = stages_list[i]
    to_stage = stages_list[i + 1]
    rate = stage_counts[to_stage] / stage_counts[from_stage] if stage_counts[from_stage] > 0 else 0
    conversion_rates[f"{from_stage} → {to_stage}"] = rate
    print(f"  {from_stage:25s} → {to_stage:20s}: {rate*100:.1f}%")

# Conversion by channel
print("\n\nPurchase conversion rate by channel:")
channel_conv = funnel_data.groupby('channel').agg(
    total_awareness=('stage_awareness', 'sum'),
    total_purchase=('stage_purchase', 'sum')
).reset_index()
channel_conv['conversion'] = channel_conv['total_purchase'] / channel_conv['total_awareness']
print(channel_conv.sort_values('conversion', ascending=False).to_string(index=False))

# ----------------------------------------------------------------------------
# SECTION 5.3 — Funnel Visualization
# ----------------------------------------------------------------------------
# Funnel bar chart — classic insurance funnel visualization
stage_names = ['Awareness', 'Consideration', 'Quote', 'Purchase', 'Onboarding']
counts = [stage_counts[s] for s in funnel_stages]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#74b9ff', '#a29bfe', '#6c5ce7', '#00b894', '#00d2d3']

bars = ax.barh(stage_names, counts, color=colors, edgecolor='white', height=0.6)

# Add data labels
for bar, count in zip(bars, counts):
    ax.annotate(f'{count:,} ({count/counts[0]*100:.1f}%)',
                xy=(count, bar.get_y() + bar.get_height()/2),
                ha='left', va='center', fontsize=11, fontweight='bold',
                xytext=(5, 0), textcoords='offset points')

# Add conversion arrows between bars
for i in range(len(counts) - 1):
    conv_pct = counts[i+1] / counts[i] * 100
    ax.annotate(f'  ↓ {conv_pct:.0f}%',
                xy=(counts[i]/2, i - 0.3), ha='center', fontsize=9,
                color='#e17055', fontweight='bold')

ax.invert_yaxis()  # Awareness at top
ax.set_title('Digital Insurance Acquisition Funnel', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Users')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('chapter9_funnel.png', dpi=120)  # artifact for submission
plt.show()

# Identify biggest drop-off point
drop_offs = []
for i in range(len(counts) - 1):
    drop = (counts[i] - counts[i+1]) / counts[i] * 100
    drop_offs.append({'from': stage_names[i], 'to': stage_names[i+1], 'drop_pct': drop})

biggest_drop = max(drop_offs, key=lambda x: x['drop_pct'])
print(f"\nBiggest drop-off: {biggest_drop['from']} → {biggest_drop['to']}")
print(f"  {biggest_drop['drop_pct']:.1f}% of users drop off at this stage")
print(f"  Focus optimization efforts here.")

# ----------------------------------------------------------------------------
# SECTION 6.1 — Channel-Level CAC and LTV Analysis
# ----------------------------------------------------------------------------
# Channel cost and acquisition data
channel_data = pd.DataFrame({
    'channel': ['organic_search', 'paid_search', 'social_media',
                'aggregator', 'referral', 'embedded_partner'],
    'monthly_spend': [200000, 1500000, 1200000, 800000, 50000, 300000],
    'new_customers': [400, 450, 350, 600, 200, 800],
    'avg_premium': [8500, 7800, 7200, 7100, 8800, 6800],
    'loss_ratio': [0.62, 0.68, 0.72, 0.70, 0.58, 0.74],
    'retention_rate': [0.80, 0.72, 0.68, 0.65, 0.85, 0.60],
    'expense_ratio': [0.10, 0.12, 0.14, 0.08, 0.10, 0.15]  # Channel-specific
})

# Calculate CAC
channel_data['cac'] = channel_data['monthly_spend'] / channel_data['new_customers']

# Calculate profit margin and annual profit per customer
channel_data['profit_margin'] = (1 - channel_data['loss_ratio']
                                  - channel_data['expense_ratio'])
channel_data['annual_profit'] = (channel_data['avg_premium']
                                  * channel_data['profit_margin'])

# Calculate LTV (simplified: annual profit × avg tenure)
channel_data['avg_tenure'] = 1 / (1 - channel_data['retention_rate'])
channel_data['ltv'] = channel_data['annual_profit'] * channel_data['avg_tenure']

# Calculate LTV/CAC ratio
channel_data['ltv_cac_ratio'] = channel_data['ltv'] / channel_data['cac']

# Sort by LTV/CAC descending
channel_data = channel_data.sort_values('ltv_cac_ratio', ascending=False)

print("=" * 100)
print(f"{'Channel':20s} {'CAC (₹)':>10s} {'LTV (₹)':>10s} {'LTV/CAC':>8s} {'Volume':>8s} {'Rating'}")
print("=" * 100)

for _, row in channel_data.iterrows():
    rating = 'EXCELLENT' if row['ltv_cac_ratio'] > 5 \
        else 'GOOD' if row['ltv_cac_ratio'] > 3 \
        else 'MARGINAL' if row['ltv_cac_ratio'] > 1.5 \
        else 'POOR (FIX OR KILL)'
    print(f"{row['channel']:20s} ₹{row['cac']:>7,.0f}  ₹{row['ltv']:>8,.0f}  {row['ltv_cac_ratio']:>5.1f}x  {row['new_customers']:>5d}  {rating}")

print("\n" + "=" * 100)
print("RECOMMENDED BUDGET ALLOCATION (based on LTV/CAC):")
print("=" * 100)

total_ltv = channel_data['ltv'].sum()
channel_data['budget_share'] = channel_data['ltv_cac_ratio'] / channel_data['ltv_cac_ratio'].sum()

for _, row in channel_data.iterrows():
    recommended = row['budget_share'] / channel_data['budget_share'].sum()
    # Show investment recommendation
    if row['ltv_cac_ratio'] > 3:
        action = 'INCREASE INVESTMENT'
    elif row['ltv_cac_ratio'] > 1.5:
        action = 'MAINTAIN'
    else:
        action = 'REDUCE OR RESTRUCTURE'
    print(f"  {row['channel']:20s}: {action} (LTV/CAC = {row['ltv_cac_ratio']:.1f}x)")

# ----------------------------------------------------------------------------
# SECTION 6.2 — Visualizing Channel Economics
# Bubble chart: CAC vs. LTV with bubble size = customer volume
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Color by LTV/CAC ratio
colors = channel_data['ltv_cac_ratio']
normalized_size = channel_data['new_customers'] / channel_data['new_customers'].max() * 800

scatter = ax.scatter(
    channel_data['cac'],
    channel_data['ltv'],
    s=normalized_size,      # Bubble size = volume
    c=colors,               # Color = LTV/CAC
    cmap='RdYlGn',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add channel labels
for _, row in channel_data.iterrows():
    label = row['channel'].replace('_', ' ').title()
    ax.annotate(label,
                (row['cac'], row['ltv']),
                fontsize=9, ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')

# Reference lines for LTV/CAC thresholds
max_val = max(channel_data['cac'].max(), channel_data['ltv'].max()) * 1.2
x = np.linspace(0, max_val)
ax.plot(x, x * 3, 'g--', alpha=0.5, label='LTV/CAC = 3x (Target)')
ax.plot(x, x * 1, 'r--', alpha=0.5, label='LTV/CAC = 1x (Break-even)')

ax.set_xlabel('CAC (₹)')
ax.set_ylabel('LTV (₹)')
ax.set_title('Channel Economics: CAC vs. LTV (Bubble Size = Volume)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Colorbar for LTV/CAC
cbar = plt.colorbar(scatter)
cbar.set_label('LTV / CAC Ratio', fontsize=10)

plt.tight_layout()
plt.savefig('channel_economics_bubble.png', dpi=120)  # artifact for submission
plt.show()

print("\nChannels above the green line (LTV/CAC > 3) are healthy growth candidates.")
print("Channels between red and green lines need monitoring and optimization.")
print("Channels below the red line (LTV/CAC < 1) destroy value — fix or restructure.")
