"""
====================================================================
 Session 06 — Data Visualization for Insurance (Management Edition)
 Complete runnable script for every chart in the session.
====================================================================
Run in Jupyter Notebook (cells) or as a plain script:
    python visualization_analysis.py

Requires: data/insurance_cleaned.csv  (produced by Session 05's
          data_analysis.py — download both from this repository)

Every section below maps to a section in the session page:
    SECTION 2 → Matplotlib foundations
    SECTION 3 → Seaborn statistical plots
    SECTION 4 → Claims trend analysis
    SECTION 5 → Portfolio composition
    SECTION 6 → Distribution & outliers
    SECTION 7 → Correlation analysis
    SECTION 8 → Publication-ready charts
    HANDS-ON  → The 6-chart board brief

Manager's lens: for each chart ask WHAT / SO WHAT / NOW WHAT.
====================================================================
"""

# ============================================================
# SECTION 2: MATPLOTLIB FOUNDATIONS
# ============================================================

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display plots in the notebook
# %matplotlib inline   # (uncomment in Jupyter)

# Set a consistent style
plt.style.use('seaborn-v0_8-darkgrid')  # clean, modern look
sns.set_palette('husl')  # distinct colors for categories

# Default figure size
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# Load the cleaned dataset from Session 05
df = pd.read_csv('data/insurance_cleaned.csv')
df['claim_date'] = pd.to_datetime(df['claim_date'])

print("Setup complete. Dataset:", df.shape)


# --- 2.2 The anatomy of a Matplotlib figure -----------------
# fig (the container) + ax (the plot). Use this pattern for all charts.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label='Example')
ax.set_title('Chart Title', fontsize=14, fontweight='bold')
ax.set_xlabel('X-Axis Label', fontsize=11)
ax.set_ylabel('Y-Axis Label', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- 2.3 Essential customizations ----------------------------
# The building blocks used throughout this session:
# ax.plot(x, y, color='#6c5ce7', linewidth=2, linestyle='-', marker='o', markersize=6)
# ax.annotate('Event', xy=(date, value), xytext=(date, value*1.2),
#             arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
# ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Target: 95%')
# ax.tick_params(axis='x', rotation=45)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
# ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


# ============================================================
# SECTION 3: SEABORN STATISTICAL PLOTS
# ============================================================

# --- 3.1 Box plot: distribution of claim amounts by policy type ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='policy_type', y='claim_amount', palette='husl')
plt.title('Distribution of Claim Amounts by Policy Type', fontweight='bold')
plt.xlabel('Policy Type')
plt.ylabel('Claim Amount (₹)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Box = interquartile range (Q1-Q3); line inside = median;
# whiskers = 1.5x IQR; points beyond = potential outliers.


# --- 3.1 Violin plot: shape & density -----------------------
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='policy_type', y='claim_amount', palette='muted')
plt.title('Claim Amount Distribution: Shape & Density', fontweight='bold')
plt.tight_layout()
plt.show()
# Wider sections = more observations at that value.


# --- 3.1 Pair plot: quick correlation overview --------------
numeric_cols = ['age', 'income', 'credit_score', 'premium', 'claim_amount', 'sum_assured']
available_cols = [c for c in numeric_cols if c in df.columns]
sns.pairplot(
    df[available_cols].dropna().sample(min(1000, len(df))),
    diag_kind='kde', corner=True
)
plt.suptitle('Pairwise Relationships: Insurance Variables', y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()
# Always sample for pair plots when n > 5000.


# --- 3.2 FacetGrid: distribution by policy_type AND status ---
g = sns.FacetGrid(
    df[df['claim_amount'].notna()].sample(min(5000, len(df))),
    col='policy_type', row='status', hue='policy_type',
    height=4, aspect=1.2, sharex=False
)
g.map(sns.histplot, 'claim_amount', bins=30, kde=True, alpha=0.6)
g.add_legend()
g.figure.suptitle('Claim Amount Distribution by Policy Type and Status', y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 4: CLAIMS TREND ANALYSIS
# ============================================================

# --- 4.1 Monthly claims trend (bars + 3-month rolling average) ---
monthly_claims = df.set_index('claim_date').resample('M').agg({
    'claim_id': 'count',
    'claim_amount': 'sum',
    'premium': 'mean'
}).rename(columns={'claim_id': 'claim_count'}).reset_index()
monthly_claims['month'] = monthly_claims['claim_date'].dt.strftime('%Y-%m')

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(monthly_claims['claim_date'], monthly_claims['claim_count'],
        color='#a29bfe', alpha=0.7, width=20, label='Monthly Claim Count')
monthly_claims['rolling_avg'] = monthly_claims['claim_count'].rolling(window=3).mean()
ax1.plot(monthly_claims['claim_date'], monthly_claims['rolling_avg'],
         color='#6c5ce7', linewidth=2.5, marker='o', markersize=5,
         label='3-Month Rolling Avg')
ax1.set_title('Monthly Insurance Claims Trend', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Claims')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

max_idx = monthly_claims['claim_count'].idxmax()
ax1.annotate(f"Peak: {int(monthly_claims.loc[max_idx, 'claim_count'])} claims",
             xy=(monthly_claims.loc[max_idx, 'claim_date'], monthly_claims.loc[max_idx, 'claim_count']),
             xytext=(monthly_claims.loc[max_idx, 'claim_date'], monthly_claims.loc[max_idx, 'claim_count'] * 1.1),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', ha='center')
plt.tight_layout()
plt.show()

print(f"Period: {monthly_claims['claim_date'].min().strftime('%b %Y')} to "
      f"{monthly_claims['claim_date'].max().strftime('%b %Y')}")
print(f"Total claims: {monthly_claims['claim_count'].sum():,}")
print(f"Monthly average: {monthly_claims['claim_count'].mean():.0f}")
print(f"Monthly std dev: {monthly_claims['claim_count'].std():.0f}")


# --- 4.2 Year-over-year comparison ---------------------------
monthly_claims['year'] = monthly_claims['claim_date'].dt.year
monthly_claims['month_num'] = monthly_claims['claim_date'].dt.month
years = sorted(monthly_claims['year'].unique())
last_two = years[-2:] if len(years) >= 2 else years

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#6c5ce7', '#00d2d3']
for i, year in enumerate(last_two):
    year_data = monthly_claims[monthly_claims['year'] == year]
    ax.plot(year_data['month_num'], year_data['claim_count'],
            color=colors[i], linewidth=2.5, marker='o', markersize=6, label=f'{year}')
    for _, row in year_data.iterrows():
        ax.annotate(str(int(row['claim_count'])),
                    (row['month_num'], row['claim_count']),
                    fontsize=8, ha='center', va='bottom', color=colors[i])

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title('Monthly Claim Count: Year-over-Year Comparison', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Claims')
ax.legend(); ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()

total_y0 = monthly_claims[monthly_claims['year'] == last_two[0]]['claim_count'].sum()
total_y1 = monthly_claims[monthly_claims['year'] == last_two[1]]['claim_count'].sum()
pct_change = ((total_y1 - total_y0) / total_y0) * 100
print(f"\n{last_two[0]} total: {total_y0:,}")
print(f"{last_two[1]} total: {total_y1:,}")
print(f"YoY change: {pct_change:+.1f}%")


# ============================================================
# SECTION 5: PORTFOLIO COMPOSITION
# ============================================================

# --- 5.1 Premium by product type (horizontal bar) -----------
product_premium = df.groupby('policy_type').agg(
    total_premium=('premium', 'sum'),
    policy_count=('policy_id', 'nunique'),
    avg_premium=('premium', 'mean')
).sort_values('total_premium', ascending=True)

print("\nPortfolio Composition:")
for product, row in product_premium.iterrows():
    share = row['total_premium'] / product_premium['total_premium'].sum() * 100
    print(f"  {product:12s} | ₹{row['total_premium']/1e7:.1f} Cr | {row['policy_count']:6,.0f} policies | {share:5.1f}% share")

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(product_premium)))
bars = ax.barh(product_premium.index, product_premium['total_premium'] / 1e7,
               color=colors, edgecolor='white', linewidth=0.5)
for bar in bars:
    ax.annotate(f"₹{bar.get_width():.1f}Cr",
                xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                ha='left', va='center', fontsize=10, fontweight='bold',
                xytext=(5, 0), textcoords='offset points')
ax.set_title('Total Premium by Product Type (₹ Crores)', fontweight='bold')
ax.set_xlabel('Premium (₹ Crores)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()


# --- 5.2 Loss ratio by product & year (grouped bar) ---------
df['claim_year'] = df['claim_date'].dt.year
loss_ratio_by_product = df.groupby(['policy_type', 'claim_year']).agg(
    total_claims=('claim_amount', 'sum'),
    total_premium=('premium', 'sum')
).reset_index()
loss_ratio_by_product['loss_ratio'] = (
    loss_ratio_by_product['total_claims'] / loss_ratio_by_product['total_premium'] * 100
)

fig, ax = plt.subplots(figsize=(14, 6))
years = sorted(loss_ratio_by_product['claim_year'].unique())
products = loss_ratio_by_product['policy_type'].unique()
colors = plt.cm.Set2(np.linspace(0, 1, len(products)))
bar_width = 0.8 / len(products)
x = np.arange(len(years))

for i, product in enumerate(products):
    product_data = loss_ratio_by_product[loss_ratio_by_product['policy_type'] == product]
    values = [product_data[product_data['claim_year'] == y]['loss_ratio'].values[0]
              if len(product_data[product_data['claim_year'] == y]) > 0 else 0 for y in years]
    offset = (i - len(products)/2 + 0.5) * bar_width
    ax.bar(x + offset, values, bar_width, label=product, color=colors[i], alpha=0.85)

ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Break-even (CR=100)')
ax.axhline(y=75, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Target (LR=75)')
ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('Loss Ratio by Product Type and Year', fontweight='bold')
ax.set_ylabel('Loss Ratio (%)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()


# --- 5.3 Channel performance (stacked bar) ------------------
channel_breakdown = df.groupby(['channel', 'policy_type']).agg(
    total_premium=('premium', 'sum')
).reset_index()
channel_pivot = channel_breakdown.pivot(
    index='channel', columns='policy_type', values='total_premium'
).fillna(0)
channel_pivot = channel_pivot.loc[channel_pivot.sum(axis=1).sort_values(ascending=True).index]

fig, ax = plt.subplots(figsize=(12, 6))
channel_pivot.plot(kind='barh', stacked=True, ax=ax, colormap='Set2',
                   edgecolor='white', linewidth=0.3)
for i, idx in enumerate(channel_pivot.index):
    total = channel_pivot.loc[idx].sum() / 1e7
    ax.annotate(f'₹{total:.1f}Cr',
                xy=(channel_pivot.loc[idx].sum(), i),
                ha='left', va='center', fontsize=9, xytext=(3, 0), textcoords='offset points')
ax.set_title('Premium Distribution by Channel and Product (₹)', fontweight='bold')
ax.set_xlabel('Total Premium')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()


# ============================================================
# SECTION 6: DISTRIBUTION & OUTLIER ANALYSIS
# ============================================================

# --- 6.1 Claim amount distribution (histogram + box) --------
claim_data = df['claim_amount'].dropna()
lower, upper = claim_data.quantile(0.01), claim_data.quantile(0.99)
claim_main = claim_data[(claim_data >= lower) & (claim_data <= upper)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(claim_main, bins=50, color='#a29bfe', edgecolor='white', alpha=0.7, density=True)
sns.kdeplot(claim_main, color='#6c5ce7', linewidth=2, ax=axes[0])
axes[0].axvline(claim_main.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f"Mean: ₹{claim_main.mean():,.0f}")
axes[0].axvline(claim_main.median(), color='green', linestyle='--', linewidth=1.5,
                label=f"Median: ₹{claim_main.median():,.0f}")
axes[0].set_title('Claim Amount Distribution (1st–99th Percentile)', fontweight='bold')
axes[0].set_xlabel('Claim Amount (₹)'); axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

sns.boxplot(data=df, x='claim_amount', color='#a29bfe', ax=axes[1],
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4, 'alpha': 0.5})
axes[1].set_title('Claim Amount Box Plot (with outliers)', fontweight='bold')
axes[1].set_xlabel('Claim Amount (₹)')
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
plt.tight_layout(); plt.show()

skew = claim_data.skew()
print(f"\nClaim Amount Statistics:")
print(f"  Observations: {len(claim_data):,}")
print(f"  Mean:    ₹{claim_data.mean():>10,.0f}")
print(f"  Median:  ₹{claim_data.median():>10,.0f}")
print(f"  Std Dev: ₹{claim_data.std():>10,.0f}")
print(f"  Skewness: {skew:.2f} {'(right-skewed)' if skew > 0 else '(left-skewed)'}")
print(f"  Top 1% of claims account for "
      f"{claim_data.sort_values(ascending=False).iloc[:int(len(claim_data)*0.01)].sum() / claim_data.sum() * 100:.1f}% of total")


# --- 6.2 Days-to-settle analysis ----------------------------
# The cleaned dataset stores settlement delay as 'days_to_settle'
# (no 'settlement_date' column), so we use it directly.
if 'days_to_settle' in df.columns:
    settle_data = df['days_to_settle'].dropna()
    settle_data = settle_data[(settle_data >= 0) & (settle_data <= 365)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(settle_data, bins=40, color='#00d2d3', edgecolor='white', alpha=0.7)
    axes[0].axvline(settle_data.mean(), color='red', linestyle='--', linewidth=1.5,
                    label=f"Mean: {settle_data.mean():.0f} days")
    axes[0].axvline(settle_data.median(), color='green', linestyle='--', linewidth=1.5,
                    label=f"Median: {settle_data.median():.0f} days")
    axes[0].set_title('Claims Settlement Time Distribution', fontweight='bold')
    axes[0].set_xlabel('Days to Settle'); axes[0].set_ylabel('Number of Claims')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    sns.boxplot(data=df[df['days_to_settle'].notna() & (df['days_to_settle'] <= 365)],
                x='policy_type', y='days_to_settle', palette='Set2', ax=axes[1])
    axes[1].set_title('Settlement Time by Product Type', fontweight='bold')
    axes[1].set_xlabel('Product Type'); axes[1].set_ylabel('Days to Settle')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()

    print(f"\nAverage settlement time: {settle_data.mean():.0f} days")
    print(f"Median settlement time:  {settle_data.median():.0f} days")
    print(f"Claims settled within 30 days: {(settle_data <= 30).mean() * 100:.1f}%")


# ============================================================
# SECTION 7: CORRELATION & RELATIONSHIP ANALYSIS
# ============================================================

# --- 7.1 Correlation heatmap --------------------------------
numeric_features = ['age', 'income', 'credit_score', 'premium',
                    'sum_assured', 'claim_amount', 'days_to_settle']
numeric_features = [c for c in numeric_features if c in df.columns]

corr_matrix = df[numeric_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}, ax=ax)
ax.set_title('Correlation Matrix — Insurance Variables', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.show()

corr_pairs = corr_matrix.unstack().dropna()
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
corr_pairs = corr_pairs.sort_values(key=abs, ascending=False)
print("\nStrongest absolute correlations:")
for (var1, var2), val in corr_pairs.head(5).items():
    direction = "positive" if val > 0 else "negative"
    print(f"  {var1:15s} ↔ {var2:15s}: {val:+.3f} ({direction})")


# --- 7.2 Scatter: premium vs. claim amount ------------------
sample = df[['premium', 'claim_amount']].dropna().sample(min(3000, len(df)))
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=sample, x='premium', y='claim_amount',
            scatter_kws={'alpha': 0.4, 's': 20, 'color': '#6c5ce7'},
            line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
corr_val = sample['premium'].corr(sample['claim_amount'])
ax.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', transform=ax.transAxes,
        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
ax.set_title('Premium vs. Claim Amount with Regression Line', fontweight='bold')
ax.set_xlabel('Premium (₹)'); ax.set_ylabel('Claim Amount (₹)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()


# --- 7.3 Multi-dimensional facet scatter --------------------
sample_multi = df[['premium', 'claim_amount', 'policy_type']].dropna().sample(min(5000, len(df)))
g = sns.FacetGrid(sample_multi, col='policy_type', col_wrap=3, height=4,
                  sharex=False, sharey=False)
g.map(sns.regplot, 'premium', 'claim_amount',
      scatter_kws={'alpha': 0.3, 's': 15}, line_kws={'color': 'red'})
g.figure.suptitle('Premium vs. Claim Amount by Policy Type', y=1.02, fontweight='bold')
plt.tight_layout(); plt.show()


# ============================================================
# SECTION 8: PUBLICATION-READY CHARTS
# ============================================================

# --- 8.2 Consistent dashboard style -------------------------
INSURANCE_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3
}
plt.rcParams.update(INSURANCE_STYLE)
INSURANCE_PALETTE = ['#6c5ce7', '#00d2d3', '#fdcb6e', '#e17055', '#00b894', '#74b9ff']


# --- 8.3 Exporting charts -----------------------------------
# After creating any chart:
# plt.savefig('claims_trend_300dpi.png', dpi=300, bbox_inches='tight', facecolor='white')  # print
# plt.savefig('claims_trend_150dpi.png', dpi=150, bbox_inches='tight', facecolor='white')  # digital
# plt.savefig('claims_trend.pdf', bbox_inches='tight', facecolor='white')                  # vector
# plt.savefig('claims_trend_ppt.png', dpi=200, bbox_inches='tight', transparent=True)      # slides


# --- 8.4 Dual-axis: premium growth vs. loss ratio -----------
yearly = df.groupby(df['claim_date'].dt.year).agg(
    total_premium=('premium', 'sum'),
    total_claims=('claim_amount', 'sum')
).reset_index()
yearly['loss_ratio'] = yearly['total_claims'] / yearly['total_premium'] * 100

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(yearly['claim_date'], yearly['total_premium'] / 1e7,
        color='#6c5ce7', alpha=0.6, width=0.6, label='Total Premium (₹ Cr)')
ax1.set_xlabel('Year'); ax1.set_ylabel('Premium (₹ Crores)', color='#6c5ce7')
ax1.tick_params(axis='y', labelcolor='#6c5ce7')
ax1.spines['top'].set_visible(False)

ax2 = ax1.twinx()
ax2.plot(yearly['claim_date'], yearly['loss_ratio'],
         color='#e17055', linewidth=2.5, marker='o', markersize=8, label='Loss Ratio (%)')
ax2.set_ylabel('Loss Ratio (%)', color='#e17055')
ax2.tick_params(axis='y', labelcolor='#e17055')
ax2.axhline(y=75, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax2.spines['top'].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.set_title('Premium Growth vs. Loss Ratio: The Sustainability Story', fontweight='bold')
plt.tight_layout(); plt.show()
# Manager's read: growing premium + stable loss ratio = healthy.
# Growing premium + rising loss ratio = "growth at any cost".


# ============================================================
# HANDS-ON: THE 6-CHART BOARD BRIEF
# ============================================================
# Chart 1 — Monthly Claims Trend: use Section 4.1 (above)
# Chart 2 — Loss Ratio by Product: use Section 5.2 (above)
# Chart 3 — Claim Distribution:    use Section 6.1 (above)
# Chart 4 — Correlation Heatmap:   use Section 7.1 (above)
# Chart 5 — Dual-Axis Growth:      use Section 8.4 (above)
# Chart 6 — Settlement by Product: use Section 6.2 (above)
#
# Then write the 1-page board brief: for each chart, two lines:
#   SO WHAT: the single business insight
#   NOW WHAT: the one decision or action it supports
# Plus a one-line Headline at the top — the single message
# the CEO must take away.
