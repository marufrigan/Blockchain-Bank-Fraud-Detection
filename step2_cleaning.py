import pandas as pd
import numpy as np
import os

os.makedirs("output", exist_ok=True)

print("=== Loading Data ===")
# Load with sample if memory issues
try:
    df = pd.read_csv("transactions.csv")
    if len(df) > 300000:
        df = df.sample(300000, random_state=42)
        print("Sampled 300,000 rows due to memory")
except:
    df = pd.read_csv("transactions.csv", nrows=300000)

accounts = pd.read_csv("account_profiles.csv")
df = df.merge(accounts, on="account_id", how="left")

print(f"Shape after merge: {df.shape}")

# Save IDs before dropping
id_cols = df[["transaction_id", "account_id"]].copy()
id_cols.reset_index(drop=True, inplace=True)
id_cols["original_index"] = id_cols.index
id_cols.to_csv("output/transaction_ids.csv", index=False)

# Save fraud_pattern before dropping for EDA
fraud_pattern_col = df["fraud_pattern"].copy() if "fraud_pattern" in df.columns else None

# Parse timestamp
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_night"] = (df["hour"].between(0, 5)).astype(int)

# Drop columns not needed for ML
drop_cols = ["transaction_id", "account_id", "timestamp", "fraud_pattern"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Encode categoricals
from sklearn.preprocessing import LabelEncoder
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].fillna("unknown")
    df[col] = le.fit_transform(df[col].astype(str))

# Fill numeric nulls
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Drop duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nFinal shape: {df.shape}")
print(f"\nFraud distribution:\n{df['is_fraud'].value_counts()}")
print(f"\nFraud rate: {df['is_fraud'].mean()*100:.2f}%")
print(f"\nNull counts: {df.isnull().sum().sum()} total nulls remaining")

df.to_csv("output/cleaned_transactions.csv", index=True, index_label="original_index")
print("\nSaved: output/cleaned_transactions.csv")

# Save fraud_pattern for EDA
if fraud_pattern_col is not None:
    fraud_pattern_col.to_csv("output/fraud_pattern_col.csv", index=False)
