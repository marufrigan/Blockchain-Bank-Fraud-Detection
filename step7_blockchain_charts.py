import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

os.makedirs("output", exist_ok=True)
TEMPLATE = "plotly_dark"

print("=== Generating Blockchain Charts ===")

ledger = pd.read_csv("output/blockchain_ledger.csv")
stats = pd.read_csv("output/blockchain_stats.csv")

print(f"Ledger shape: {ledger.shape}")
print(ledger["decision"].value_counts())

# Chart 1 — Blockchain Decisions Pie Chart
decision_counts = ledger["decision"].value_counts().reset_index()
decision_counts.columns = ["decision", "count"]

color_map = {
    "APPROVE": "#00CC96",
    "REVIEW": "#FFA500",
    "FLAG": "#EF553B",
    "REJECT": "#8B0000",
    "ERROR": "#808080"
}

fig = px.pie(
    decision_counts,
    values="count",
    names="decision",
    title="Blockchain Smart Contract Decision Distribution",
    color="decision",
    color_discrete_map=color_map,
    template=TEMPLATE
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.write_image("output/blockchain_decisions_pie.png", width=1000, height=700, scale=2)
print("Saved blockchain_decisions_pie.png")

# Chart 2 — Fraud Score by Decision Box Plot
fig = px.box(
    ledger,
    x="decision",
    y="fraud_score",
    color="decision",
    title="ML Fraud Score Distribution by Blockchain Decision",
    color_discrete_map=color_map,
    template=TEMPLATE,
    points="outliers"
)
fig.update_layout(
    xaxis_title="Blockchain Decision",
    yaxis_title="XGBoost Fraud Score",
    showlegend=False
)
fig.write_image("output/blockchain_score_by_decision.png", width=1200, height=700, scale=2)
print("Saved blockchain_score_by_decision.png")

# Chart 3 — Cumulative Fraud Caught Over Time
ledger["is_caught"] = ledger["decision"].isin(["FLAG", "REJECT"]).astype(int)
ledger["cumulative_caught"] = ledger["is_caught"].cumsum()
ledger["transaction_number"] = range(1, len(ledger) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ledger["transaction_number"],
    y=ledger["cumulative_caught"],
    mode="lines",
    fill="tozeroy",
    line=dict(color="#EF553B", width=3),
    fillcolor="rgba(239, 85, 59, 0.2)",
    name="Cumulative Fraud Caught"
))
fig.update_layout(
    title="Cumulative Fraud Flagged/Rejected by Blockchain Smart Contract",
    xaxis_title="Transaction Number",
    yaxis_title="Cumulative Fraud Caught",
    template=TEMPLATE
)
total_caught = ledger["is_caught"].sum()
fig.add_annotation(
    x=len(ledger), y=total_caught,
    text=f"Total: {total_caught} caught",
    showarrow=True, arrowhead=2,
    font=dict(color="white", size=14)
)
fig.write_image("output/blockchain_cumulative_flags.png", width=1200, height=700, scale=2)
print("Saved blockchain_cumulative_flags.png")

# Print Final Summary
print("\n" + "="*50)
print("FINAL RESEARCH PAPER SUMMARY")
print("="*50)
print(f"Total transactions in blockchain: {len(ledger)}")
for decision in ["APPROVE", "REVIEW", "FLAG", "REJECT"]:
    count = len(ledger[ledger['decision'] == decision])
    pct = count / len(ledger) * 100
    print(f"  {decision}: {count} ({pct:.1f}%)")
frozen = ledger[ledger['decision'] == 'FLAG']['account_id'].nunique()
print(f"  Unique accounts frozen: {frozen}")
print("="*50)
print("\nAll output files saved to /output folder:")
import os
for f in sorted(os.listdir("output")):
    print(f"  output/{f}")
