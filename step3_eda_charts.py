import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

os.makedirs("output", exist_ok=True)

df_clean = pd.read_csv("output/cleaned_transactions.csv")
df_original = pd.read_csv("transactions.csv")
if len(df_original) > 300000:
    df_original = df_original.sample(300000, random_state=42)

TEMPLATE = "plotly_dark"
W, H = 1200, 700

# Chart 1 — Fraud vs Legitimate Pie Chart
counts = df_clean["is_fraud"].value_counts()
fig = px.pie(
    values=counts.values,
    names=["Legitimate", "Fraud"],
    title="Fraud vs Legitimate Transaction Distribution",
    color_discrete_sequence=["#00CC96", "#EF553B"],
    template=TEMPLATE
)
fig.write_image("output/chart1_fraud_distribution_pie.png", width=W, height=H, scale=2)
print("Saved chart1")

# Chart 2 — Fraud by Pattern
if "fraud_pattern" in df_original.columns:
    pattern_counts = df_original["fraud_pattern"].dropna().value_counts().reset_index()
    pattern_counts.columns = ["fraud_pattern", "count"]
    fig = px.bar(pattern_counts, x="fraud_pattern", y="count",
                 title="Fraud Count by Fraud Pattern Type",
                 color="fraud_pattern", template=TEMPLATE,
                 labels={"fraud_pattern": "Fraud Pattern", "count": "Count"})
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    fig.write_image("output/chart2_fraud_by_pattern.png", width=W, height=H, scale=2)
    print("Saved chart2")

# Chart 3 — Fraud Rate by Merchant Category
if "merchant_category" in df_original.columns:
    df_temp = df_original[["merchant_category", "is_fraud"]].dropna()
    fraud_rate = df_temp.groupby("merchant_category")["is_fraud"].mean().reset_index()
    fraud_rate.columns = ["merchant_category", "fraud_rate"]
    fraud_rate["fraud_rate_pct"] = fraud_rate["fraud_rate"] * 100
    fraud_rate = fraud_rate.sort_values("fraud_rate_pct", ascending=False)
    fig = px.bar(fraud_rate, x="merchant_category", y="fraud_rate_pct",
                 title="Fraud Rate % by Merchant Category",
                 color="fraud_rate_pct", color_continuous_scale="Reds",
                 template=TEMPLATE)
    fig.update_layout(xaxis_tickangle=-30)
    fig.write_image("output/chart3_fraud_by_merchant.png", width=W, height=H, scale=2)
    print("Saved chart3")

# Chart 4 — Hourly Fraud Rate
if "hour" in df_clean.columns:
    hourly = df_clean.groupby("hour")["is_fraud"].mean().reset_index()
    hourly["fraud_rate_pct"] = hourly["is_fraud"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["fraud_rate_pct"],
                             mode="lines+markers", line=dict(color="#EF553B", width=3)))
    fig.add_vrect(x0=0, x1=5, fillcolor="red", opacity=0.15,
                  annotation_text="High Risk (00:00-05:00)")
    fig.update_layout(title="Fraud Rate by Hour of Day",
                      xaxis_title="Hour", yaxis_title="Fraud Rate %",
                      template=TEMPLATE)
    fig.write_image("output/chart4_fraud_by_hour.png", width=W, height=H, scale=2)
    print("Saved chart4")

# Chart 5 — Correlation Heatmap
num_df = df_clean.select_dtypes(include=[np.number])
top_cols = num_df.var().nlargest(15).index.tolist()
if "is_fraud" not in top_cols:
    top_cols = top_cols[:14] + ["is_fraud"]
corr = num_df[top_cols].corr()
fig = px.imshow(corr, title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r", template=TEMPLATE,
                text_auto=".2f", aspect="auto")
fig.write_image("output/chart5_correlation_heatmap.png", width=1400, height=900, scale=2)
print("Saved chart5")

# Chart 6 — Amount Box Plot
if "amount" in df_original.columns:
    df_box = df_original[["amount", "is_fraud"]].copy()
    cap = df_box["amount"].quantile(0.99)
    df_box = df_box[df_box["amount"] <= cap]
    df_box["Category"] = df_box["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
    fig = px.box(df_box, x="Category", y="amount",
                 title="Transaction Amount Distribution: Fraud vs Legitimate",
                 color="Category",
                 color_discrete_map={"Legitimate": "#00CC96", "Fraud": "#EF553B"},
                 template=TEMPLATE)
    fig.write_image("output/chart6_amount_boxplot.png", width=W, height=H, scale=2)
    print("Saved chart6")

# Chart 7 — Fraud Rate by Device Type
if "device_type" in df_original.columns:
    dt_fraud = df_original.groupby("device_type")["is_fraud"].mean().reset_index()
    dt_fraud["fraud_rate_pct"] = dt_fraud["is_fraud"] * 100
    dt_fraud = dt_fraud.sort_values("fraud_rate_pct", ascending=False)
    fig = px.bar(dt_fraud, x="device_type", y="fraud_rate_pct",
                 title="Fraud Rate % by Device Type",
                 color="device_type", template=TEMPLATE)
    fig.update_layout(showlegend=False)
    fig.write_image("output/chart7_fraud_by_device.png", width=W, height=H, scale=2)
    print("Saved chart7")

# Chart 8 — Fraud Rate by Foreign Transaction
if "is_foreign_txn" in df_original.columns:
    ft = df_original.groupby("is_foreign_txn")["is_fraud"].mean().reset_index()
    ft["Label"] = ft["is_foreign_txn"].map({0: "Domestic", 1: "Foreign"})
    ft["fraud_rate_pct"] = ft["is_fraud"] * 100
    fig = px.bar(ft, x="Label", y="fraud_rate_pct",
                 title="Fraud Rate %: Foreign vs Domestic Transactions",
                 color="Label",
                 color_discrete_sequence=["#636EFA", "#EF553B"],
                 template=TEMPLATE)
    fig.write_image("output/chart8_fraud_by_foreign.png", width=W, height=H, scale=2)
    print("Saved chart8")

# Chart 9 — Fraud Rate by 2FA
if "has_2fa" in df_original.columns:
    tfa = df_original.groupby("has_2fa")["is_fraud"].mean().reset_index()
    tfa["Label"] = tfa["has_2fa"].map({0: "No 2FA", 1: "Has 2FA"})
    tfa["fraud_rate_pct"] = tfa["is_fraud"] * 100
    fig = px.bar(tfa, x="Label", y="fraud_rate_pct",
                 title="Fraud Rate %: 2FA Enabled vs Disabled",
                 color="Label",
                 color_discrete_sequence=["#EF553B", "#00CC96"],
                 template=TEMPLATE)
    fig.write_image("output/chart9_fraud_by_2fa.png", width=W, height=H, scale=2)
    print("Saved chart9")

# Chart 10 — Fraud Rate by Account Age
if "account_age_days" in df_original.columns:
    df_age = df_original[["account_age_days", "is_fraud"]].copy()
    df_age["age_bin"] = pd.qcut(df_age["account_age_days"], q=10, duplicates="drop")
    age_fraud = df_age.groupby("age_bin", observed=True)["is_fraud"].mean().reset_index()
    age_fraud["fraud_rate_pct"] = age_fraud["is_fraud"] * 100
    age_fraud["age_bin_str"] = age_fraud["age_bin"].astype(str)
    fig = px.line(age_fraud, x="age_bin_str", y="fraud_rate_pct",
                  title="Fraud Rate by Account Age (binned)",
                  markers=True, template=TEMPLATE)
    fig.update_layout(xaxis_tickangle=-30)
    fig.write_image("output/chart10_fraud_by_account_age.png", width=W, height=H, scale=2)
    print("Saved chart10")

print("\n=== All EDA Charts Saved to /output ===")
