# Core data manipulation libraries
import pandas as pd
import numpy as np

# Optional: display all columns and rows in Jupyter
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# For reproducibility
np.random.seed(42)

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# Specify the data path
DATA_PATH = 'data/'

# Load each table
customers   = pd.read_csv(DATA_PATH + 'customers.csv')
policies    = pd.read_csv(DATA_PATH + 'policies.csv')
claims      = pd.read_csv(DATA_PATH + 'claims.csv')
fraud_indicators = pd.read_csv(DATA_PATH + 'fraud_indicators.csv')
cyber       = pd.read_csv(DATA_PATH + 'cyber_incidents.csv')
weather     = pd.read_csv(DATA_PATH + 'weather_data.csv')

# Confirm all tables loaded
print("Tables loaded successfully:")
print(f"  Customers:       {customers.shape[0]:,} rows")
print(f"  Policies:        {policies.shape[0]:,} rows")
print(f"  Claims:          {claims.shape[0]:,} rows")
print(f"  Fraud Indicators:{fraud_indicators.shape[0]:,} rows")
print(f"  Cyber Incidents: {cyber.shape[0]:,} rows")
print(f"  Weather:         {weather.shape[0]:,} rows")

# Examine the customers table
print("=== CUSTOMERS TABLE ===")
print(f"Shape: {customers.shape}")
print(f"\nColumn names and types:\n{customers.dtypes}")
print(f"\nFirst 5 rows:\n{customers.head()}")
print(f"\nSummary statistics (numeric):\n{customers.describe()}")
print(f"\nSummary statistics (categorical):\n{customers.describe(include='object')}")

# Check for missing values in each column
print("=== MISSING VALUES ===")
print(customers.isnull().sum())
print(f"\nPercentage missing:\n{customers.isnull().mean() * 100}")

# Unique values in categorical columns
print(f"\nUnique values in 'occupation': {customers['occupation'].nunique()}")
print(f"Unique values in 'location': {customers['location'].nunique()}")
print(f"Value counts for 'gender':\n{customers['gender'].value_counts()}")

# Drop rows where ANY value is missing (use with extreme caution)
df_clean = customers.dropna()

# Drop rows where a SPECIFIC column is missing (safer)
df_clean = customers.dropna(subset=['customer_id', 'age'])

# Drop columns with more than 50% missing values
threshold = len(customers) * 0.5
customers = customers.dropna(axis=1, thresh=threshold)

# Convert date columns from string to datetime
policies['start_date'] = pd.to_datetime(policies['start_date'])
policies['end_date']   = pd.to_datetime(policies['end_date'])
claims['claim_date']   = pd.to_datetime(claims['claim_date'])

# Verify the conversion
print(policies[['start_date', 'end_date']].dtypes)
print(f"Date range: {policies['start_date'].min()} to {policies['start_date'].max()}")

# Create derived date features
claims['claim_year']    = claims['claim_date'].dt.year
claims['claim_month']   = claims['claim_date'].dt.month
claims['claim_quarter'] = claims['claim_date'].dt.quarter
claims['day_of_week']   = claims['claim_date'].dt.dayofweek  # 0=Monday, 6=Sunday

# Check if premium column is actually numeric
print(f"Premium dtype: {policies['premium'].dtype}")
print(f"Unique non-numeric values:\n{policies[pd.to_numeric(policies['premium'], errors='coerce').isna()]['premium'].unique()}")

if policies['premium'].dtype == 'object':
    policies['premium'] = policies['premium']\
        .str.replace('₹', '')\
        .str.replace(',', '')\
        .str.strip()\
        .astype(float)

# For claim amounts — check for outliers first using .describe()
print(claims['claim_amount'].describe())

# Cap extreme outliers at the 99th percentile (optional, based on business context)
cap = claims['claim_amount'].quantile(0.99)
claims['claim_amount_capped'] = claims['claim_amount'].clip(upper=cap)

# Check unique values before cleaning
print("Before cleaning:", policies['policy_type'].unique())

# Standardize to title case
policies['policy_type'] = policies['policy_type'].str.strip().str.title()

# Map variations to standard categories
type_mapping = {
    'Motor Own Damage': 'Motor',
    'Motor Od': 'Motor',
    'Motor': 'Motor',
    'Health': 'Health',
    'Health Insurance': 'Health',
    'Property': 'Property',
    'Property Insurance': 'Property',
    'Fire': 'Property',
    'Crop': 'Crop',
    'Crop Insurance': 'Crop',
    'Travel': 'Travel'
}
policies['policy_type'] = policies['policy_type'].map(type_mapping)

print("After cleaning:", policies['policy_type'].unique())
print(f"Value counts:\n{policies['policy_type'].value_counts()}")

# Check for exact duplicates across all columns
exact_dups = claims.duplicated().sum()
print(f"Exact duplicate rows: {exact_dups}")

# Check for duplicates based on key columns (more common in insurance)
# A claim_id should be unique — any duplicate is a data error
dup_claim_ids = claims['claim_id'].duplicated().sum()
print(f"Duplicate claim IDs: {dup_claim_ids}")

# Show the actual duplicates (if any)
if dup_claim_ids > 0:
    dupe_mask = claims['claim_id'].duplicated(keep=False)
    print(claims[dupe_mask].sort_values('claim_id').head(10))

# Step 1: Merge Customer → Policy (left join: keep all policies, add customer info)
policy_customer = policies.merge(
    customers,
    on='customer_id',
    how='left'  # Keep all policies even if customer data is missing
)

# Step 2: Merge Policy+Customer → Claims
full_data = claims.merge(
    policy_customer,
    on=['policy_id', 'customer_id'],
    how='left'  # Keep all claims
)

# Verify the merged result
print(f"Merged dataset shape: {full_data.shape}")
print(f"Columns: {full_data.columns.tolist()}")
print(f"Missing values in merged dataset:\n{full_data.isnull().sum()}")

# Optional: merge fraud indicators
full_data = full_data.merge(
    fraud_indicators,
    on='claim_id',
    how='left'
)

print(f"Final dataset shape: {full_data.shape}")

# Check: did every claim get matched to a policy?
claims_unmatched = full_data[full_data['premium'].isna()].shape[0]
print(f"Claims without matching policy: {claims_unmatched}")

# Check: did every policy get matched to a customer?
policies_unmatched = policy_customer[policy_customer['age'].isna()]['policy_id'].nunique()
print(f"Policies without matching customer: {policies_unmatched}")

# Row count sanity check
print(f"Original claims: {len(claims)}")
print(f"Merged dataset:  {len(full_data)}")
if len(full_data) > len(claims):
    print("CAUTION: Merge created extra rows — check for duplicate keys!")
elif len(full_data) < len(claims):
    print("CAUTION: Merge dropped rows — check that all claim IDs exist in policies table!")


def generate_data_quality_report(df, dataset_name):
    """
    Generate a comprehensive data quality report for a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame — The dataset to report on
    dataset_name : str — Name of the dataset for the report header

    Returns:
    --------
    pd.DataFrame — A quality report with one row per column
    """
    report = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.notna().sum().values,
        'Null Count': df.isna().sum().values,
        'Null %': (df.isna().mean() * 100).values,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Has Duplicates': [df[col].duplicated().any() for col in df.columns],
    })

    # Detect numeric columns for outlier check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['Has Outliers (IQR)'] = '—'

    for col in df.columns:
        if col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            report.loc[report['Column'] == col, 'Has Outliers (IQR)'] = \
                'Yes (' + str(outlier_count) + ')' if outlier_count > 0 else 'No'

    # Sort by null percentage descending
    report = report.sort_values('Null %', ascending=False).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT: {dataset_name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Columns with nulls: {(df.isna().sum() > 0).sum()} of {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"{'='*60}\n")

    return report


# Generate reports for each table
for name, df in [
    ('Customers', customers),
    ('Policies', policies),
    ('Claims', claims),
    ('Merged Dataset', full_data)
]:
    report = generate_data_quality_report(df, name)
    try:
        display(report)
    except NameError:
        print(report)
    print("\n")

# Export the cleaned, merged dataset
full_data.to_csv('data/insurance_cleaned.csv', index=False)
print(f"Cleaned dataset saved: data/insurance_cleaned.csv ({len(full_data):,} rows)")

# Also save individual cleaned tables
customers.to_csv('data/customers_cleaned.csv', index=False)
policies.to_csv('data/policies_cleaned.csv', index=False)
claims.to_csv('data/claims_cleaned.csv', index=False)

# Save the quality report for reference
quality_report = generate_data_quality_report(full_data, 'Merged Dataset')
quality_report.to_csv('data/data_quality_report.csv', index=False)
print("Data quality report saved: data/data_quality_report.csv")