# ============================================================================
# Session 11 — AI & ML Fundamentals (COMPLETE RUNNABLE SCRIPT)
# InsurTech & Digital Risk Solutions (MBA) — Woxsen University
#
# HOW TO RUN:
#   venv/bin/python session_11_ml_fundamentals.py
#   (or run top-to-bottom in one Jupyter notebook — the page's §4–§7 blocks
#    are merged here in order so nothing is left undefined)
#
# DATA FILE: data/insurance_cleaned.csv (provided — merged/cleaned claims +
#            policies + customers + fraud indicators). Falls back to
#            insurance_cleaned.csv in the current folder.
#
# WHAT IT COVERS (in order):
#   §4  Regression — predicting claim amounts
#   §5  Classification — predicting claim probability
#   §6  Clustering — segmenting policyholders
#   §7  Overfitting & cross-validation
#
# OUTPUTS: console tables + regression_diagnostics.png + roc_curve.png +
#          elbow_curve.png (saved alongside the console output)
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

# ----------------------------------------------------------------------------
# SECTION 4.1 — Preparing Data for Regression
# ----------------------------------------------------------------------------
try:
    df = pd.read_csv('data/insurance_cleaned.csv')
except FileNotFoundError:
    df = pd.read_csv('insurance_cleaned.csv')

print(f"Loaded insurance_cleaned.csv: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Filter to claims with known amounts (non-null)
reg_data = df[df['claim_amount'].notna()].copy()

# Define features for predicting claim amount
feature_cols = ['age', 'income', 'credit_score', 'premium',
                'sum_assured', 'policy_type', 'vehicle_age' if 'vehicle_age' in df.columns else None]
feature_cols = [c for c in feature_cols if c is not None and c in reg_data.columns]

# Drop rows with missing feature values (income has some nulls in the cleaned file)
reg_data = reg_data.dropna(subset=feature_cols)

categorical_cols = reg_data[feature_cols].select_dtypes(include=['object']).columns.tolist()
numeric_cols = reg_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"Numeric features: {numeric_cols}")
print(f"Categorical features: {categorical_cols}")
print(f"Total observations: {len(reg_data):,}")

# ----------------------------------------------------------------------------
# SECTION 4.2 — Building the Regression Pipeline
# ----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_reg = reg_data[feature_cols]
y_reg = reg_data['claim_amount']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_pipeline.fit(X_train_r, y_train_r)
y_pred_r = reg_pipeline.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("=" * 50)
print("CLAIM AMOUNT REGRESSION RESULTS")
print("=" * 50)
print(f"RMSE:  ₹{rmse:,.0f}")
print(f"MAE:   ₹{mae:,.0f}")
print(f"R²:    {r2:.3f}")
print(f"\nInterpretation:")
print(f"  The model predicts claim amounts with an average error of ₹{mae:,.0f}")
print(f"  R²={r2:.3f} means the model explains {r2*100:.1f}% of the variance in claim amounts")
print(f"  (Context: average claim = ₹{y_test_r.mean():,.0f})")
print(f"  The prediction error represents {mae/y_test_r.mean()*100:.1f}% of the average claim")

# ----------------------------------------------------------------------------
# SECTION 4.3 — Interpreting Coefficients
# ----------------------------------------------------------------------------
lr_model = reg_pipeline.named_steps['regressor']
preprocessor = reg_pipeline.named_steps['preprocessor']
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist() if len(categorical_cols) > 0 else []
all_feature_names = numeric_cols + cat_feature_names

coeff_df = pd.DataFrame({
    'feature': all_feature_names,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Most Influential Features (by absolute coefficient):")
print("=" * 60)
print(f"{'Feature':30s} {'Coefficient':>15s} {'Direction':>12s}")
print("-" * 60)
for _, row in coeff_df.head(10).iterrows():
    direction = "Increases claim" if row['coefficient'] > 0 else "Decreases claim"
    print(f"{row['feature']:30s} ₹{row['coefficient']:>+8,.0f}   {direction}")

print(f"\n{'-' * 60}")
print(f"Intercept: ₹{lr_model.intercept_:,.0f}")
print(f"\nNote: Coefficients show the change in predicted claim amount for a")
print(f"one-unit increase in the feature (with all other features held constant).")

# ----------------------------------------------------------------------------
# SECTION 4.4 — Diagnostic Plot
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_r, y_pred_r, alpha=0.3, s=10, color='#6c5ce7')
axes[0].plot([y_test_r.min(), y_test_r.max()],
             [y_test_r.min(), y_test_r.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Claim Amount')
axes[0].set_ylabel('Predicted Claim Amount')
axes[0].set_title('Regression: Actual vs. Predicted', fontweight='bold')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

residuals = y_test_r - y_pred_r
axes[1].scatter(y_pred_r, residuals, alpha=0.3, s=10, color='#00d2d3')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Predicted Claim Amount')
axes[1].set_ylabel('Residual (Actual − Predicted)')
axes[1].set_title('Residual Plot', fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('regression_diagnostics.png', dpi=120)
plt.show()

print("\nResidual analysis:")
print(f"  Mean residual: {residuals.mean():.0f} (should be close to 0)")
print(f"  Std of residuals: {residuals.std():.0f}")
print("  (A cone-shaped or curved residual pattern would indicate the linear model is inadequate.)")

# ----------------------------------------------------------------------------
# SECTION 5.1 — Preparing Data for Classification
# ----------------------------------------------------------------------------
# Load a POLICY-LEVEL dataset (one row per policy, including no-claim policies).
# The claim-level df above cannot produce a "no-claim" class (every row is a
# claim), so classification must start from the policy table with has_claim.
try:
    policy_df = pd.read_csv('data/insurance_policy_cleaned.csv')
except FileNotFoundError:
    policy_df = pd.read_csv('insurance_policy_cleaned.csv')

claim_rate = policy_df['has_claim'].mean()
print("\n" + "=" * 50)
print("CLASSIFICATION — TARGET SETUP")
print("=" * 50)
print(f"Claim rate: {claim_rate*100:.1f}%")
print(f"  Claim filed (1): {(policy_df['has_claim'] == 1).sum():,}")
print(f"  No claim (0):    {(policy_df['has_claim'] == 0).sum():,}")
print(f"  Ratio: 1:{((1 - claim_rate) / claim_rate):.1f}")

class_features = ['premium', 'age', 'income', 'credit_score', 'sum_assured', 'policy_type']
class_features = [c for c in class_features if c in policy_df.columns]

# Drop rows with missing feature values (income has some nulls)
policy_df = policy_df.dropna(subset=class_features)
X_cls = policy_df[class_features]
y_cls = policy_df['has_claim']

# ----------------------------------------------------------------------------
# SECTION 5.2 — Building the Classification Pipeline
# ----------------------------------------------------------------------------
cat_cls = X_cls.select_dtypes(include=['object']).columns.tolist()
num_cls = X_cls.select_dtypes(include=[np.number]).columns.tolist()

preprocessor_cls = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cls),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cls)
    ])

cls_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_cls),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

cls_pipeline.fit(X_train_c, y_train_c)
y_pred_c = cls_pipeline.predict(X_test_c)
y_prob_c = cls_pipeline.predict_proba(X_test_c)[:, 1]

print("\n" + "=" * 60)
print("CLAIM PROBABILITY CLASSIFICATION RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test_c, y_pred_c):.3f}")
print(f"Precision: {precision_score(y_test_c, y_pred_c):.3f}")
print(f"Recall:    {recall_score(y_test_c, y_pred_c):.3f}")
print(f"F1 Score:  {f1_score(y_test_c, y_pred_c):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_c, y_prob_c):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=['No Claim', 'Claim']))

cm = confusion_matrix(y_test_c, y_pred_c)
print(f"\nConfusion Matrix:")
print(f"            Predicted No    Predicted Yes")
print(f"Actual No   {cm[0,0]:>6,d}        {cm[0,1]:>6,d}")
print(f"Actual Yes  {cm[1,0]:>6,d}        {cm[1,1]:>6,d}")

# ----------------------------------------------------------------------------
# SECTION 5.4 — ROC Curve
# ----------------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test_c, y_prob_c)
auc = roc_auc_score(y_test_c, y_prob_c)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#6c5ce7', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Classifier (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.15, color='#6c5ce7')
ax.set_xlabel('False Positive Rate (1 − Specificity)')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve — Claim Prediction Model', fontweight='bold')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
ax.annotate(f'Optimal threshold: {best_threshold:.2f}',
            xy=(fpr[best_idx], tpr[best_idx]),
            xytext=(fpr[best_idx] + 0.15, tpr[best_idx] - 0.1),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')
ax.plot(fpr[best_idx], tpr[best_idx], 'go', markersize=10)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=120)
plt.show()

print(f"\nOptimal threshold (Youden's J): {best_threshold:.3f}")
print(f"  At this threshold: Sensitivity = {tpr[best_idx]:.3f}, Specificity = {1-fpr[best_idx]:.3f}")
print(f"\nInterpretation: A model with AUC = {auc:.3f} can distinguish")
print(f"claim-filers from non-claim-filers {auc*100:.1f}% of the time.")

# ----------------------------------------------------------------------------
# SECTION 6.2 — K-Means Clustering in Python
# ----------------------------------------------------------------------------
# Build a CUSTOMER-LEVEL profile from the policy-level table so that customers
# with no claims are included (unlike aggregating the claim-level df).
customer_profile = policy_df.groupby('customer_id').agg(
    age=('age', 'first'),
    income=('income', 'first'),
    credit_score=('credit_score', 'first'),
    total_premium=('premium', 'sum'),
    total_claims=('total_claim_amount', 'sum'),
    claim_count=('n_claims', 'sum'),
    policy_count=('policy_id', 'nunique')
).reset_index()

customer_profile['loss_ratio'] = (
    customer_profile['total_claims'] / customer_profile['total_premium']
).fillna(0)
customer_profile['claims_per_policy'] = (
    customer_profile['claim_count'] / customer_profile['policy_count']
)

clust_cols = ['age', 'income', 'credit_score', 'total_premium', 'loss_ratio',
              'claims_per_policy', 'policy_count']
clust_cols = [c for c in clust_cols if c in customer_profile.columns]

X_clust = customer_profile[clust_cols].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'o-', color='#6c5ce7', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax.set_title('Elbow Method for Optimal K', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('Elbow point', xy=(4, inertias[2]),
            xytext=(5, inertias[2] + inertias[0]*0.1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, color='red')
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=120)
plt.show()

print("\nElbow Analysis:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: Inertia = {inertia:,.0f}")

# ----------------------------------------------------------------------------
# SECTION 6.3 — Profiling the Clusters
# ----------------------------------------------------------------------------
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
customer_profile['cluster'] = kmeans.fit_predict(X_scaled)

cluster_profile = customer_profile.groupby('cluster').agg(
    count=('customer_id', 'count'),
    avg_age=('age', 'mean'),
    avg_income=('income', 'mean'),
    avg_credit=('credit_score', 'mean'),
    avg_premium=('total_premium', 'mean'),
    avg_loss_ratio=('loss_ratio', 'mean'),
    avg_claims_per_policy=('claims_per_policy', 'mean'),
    avg_policies=('policy_count', 'mean')
).round(1)

cluster_profile = cluster_profile.sort_values('count', ascending=False)
print("\n" + "=" * 90)
print("CUSTOMER SEGMENTATION — CLUSTER PROFILES")
print("=" * 90)
print(cluster_profile.to_string())

print(f"\n{'='*90}")
print("BUSINESS INTERPRETATION:")
print(f"{'='*90}")
for cluster_id in cluster_profile.index:
    row = cluster_profile.loc[cluster_id]
    share = row['count'] / cluster_profile['count'].sum() * 100
    if row['avg_loss_ratio'] < 0.3 and row['avg_income'] > cluster_profile['avg_income'].median():
        label = "Low-Risk, High-Value"
    elif row['avg_loss_ratio'] > 0.6:
        label = "High-Risk — Manage Carefully"
    elif row['avg_claims_per_policy'] < 0.1 and row['avg_policies'] > 2:
        label = "Loyal, Low-Claim — Protect"
    elif row['avg_age'] > 50:
        label = "Senior Segment — Growing"
    else:
        label = "Standard Risk — Optimize Service"
    print(f"  Cluster {cluster_id} ({share:.0f}% of customers): {label}")
    print(f"    Age: {row['avg_age']:.0f}, Income: ₹{row['avg_income']:,.0f}, Credit: {row['avg_credit']:.0f}")
    print(f"    Loss Ratio: {row['avg_loss_ratio']*100:.1f}%, Policies: {row['avg_policies']:.1f}, Claims/Policy: {row['avg_claims_per_policy']:.2f}")
    print()

# ----------------------------------------------------------------------------
# SECTION 7.2 — Overfitting Demonstration
# ----------------------------------------------------------------------------
print("=" * 55)
print("OVERFITTING DEMONSTRATION — Decision Tree (no depth limit)")
print("=" * 55)
# The tree also needs categorical encoding, so reuse the preprocessor via a pipeline.
overfit_tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, random_state=42)
overfit_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('tree', overfit_tree)])
overfit_pipe.fit(X_train_r, y_train_r)
train_score = overfit_pipe.score(X_train_r, y_train_r)
test_score = overfit_pipe.score(X_test_r, y_test_r)
print(f"  Training R²:  {train_score:.4f}  {'(unrealistically high — memorized training data)' if train_score > 0.95 else ''}")
print(f"  Test R²:      {test_score:.4f}")
print(f"  Gap:          {train_score - test_score:.4f}")

regularized_tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)
regularized_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('tree', regularized_tree)])
regularized_pipe.fit(X_train_r, y_train_r)
train_score_r = regularized_pipe.score(X_train_r, y_train_r)
test_score_r = regularized_pipe.score(X_test_r, y_test_r)
print(f"\nREGULARIZED DECISION TREE (max_depth=5, min_samples_leaf=20)")
print(f"  Training R²:  {train_score_r:.4f}")
print(f"  Test R²:      {test_score_r:.4f}")
print(f"  Gap:          {train_score_r - test_score_r:.4f}")
print("  (A smaller train/test gap means better generalization.)")

# ----------------------------------------------------------------------------
# SECTION 7.3 — Cross-Validation
# ----------------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(reg_pipeline, X_reg, y_reg, cv=cv, scoring='r2')

print("\nCROSS-VALIDATION RESULTS (5-Fold)")
print("=" * 40)
for i, s in enumerate(cv_scores, 1):
    print(f"  Fold {i} R²: {s:.3f}")
print(f"  {'─' * 25}")
print(f"  Mean R²:   {cv_scores.mean():.3f}")
print(f"  Std Dev:   {cv_scores.std():.3f}")
print("\n(Done — charts saved: regression_diagnostics.png, roc_curve.png, elbow_curve.png)")
