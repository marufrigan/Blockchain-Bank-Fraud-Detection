import pandas as pd
import plotly.graph_objects as go
import os

os.makedirs("output", exist_ok=True)
TEMPLATE = "plotly_dark"

print("=== Generating Final Summary Table ===")

# Load results
fixed = pd.read_csv("output/fixed_model_results_summary.csv")

# Create a clean formatted table image for the paper
fig = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Model</b>", "<b>Accuracy</b>", "<b>Precision</b>",
                "<b>Recall</b>", "<b>F1 Score</b>", "<b>ROC-AUC</b>", "<b>PR-AUC</b>"],
        fill_color="#1f2937",
        font=dict(color="white", size=13),
        align="center",
        height=40
    ),
    cells=dict(
        values=[
            fixed["Model"],
            fixed["Accuracy"].map("{:.4f}".format),
            fixed["Precision"].map("{:.4f}".format),
            fixed["Recall"].map("{:.4f}".format),
            fixed["F1"].map("{:.4f}".format),
            fixed["ROC_AUC"].map("{:.4f}".format),
            fixed["PR_AUC"].map("{:.4f}".format),
        ],
        fill_color=[["#111827", "#1a2332", "#111827"]],
        font=dict(color="white", size=12),
        align="center",
        height=35
    )
)])
fig.update_layout(
    title="Model Performance Summary — Leakage-Free Evaluation (GroupShuffleSplit)",
    template=TEMPLATE,
    margin=dict(l=20, r=20, t=60, b=20)
)
fig.write_image("output/final_model_results_table.png", width=1400, height=300, scale=2)
print("Saved final_model_results_table.png")

# Print what to write in paper
print("\n=== COPY THIS INTO YOUR PAPER ===")
print("""
Evaluation Setup: All models were evaluated using GroupShuffleSplit (test_size=0.2, 
random_state=42) partitioned by account_id, ensuring no account appears in both 
train and test sets. Label-derived account-level features (fraud_rate, risk_score, 
is_high_risk) were excluded to prevent optimistic bias. Due to the class imbalance 
(~1.71% fraud rate), PR-AUC and F1 Score are used as primary evaluation metrics 
rather than ROC-AUC or accuracy.
""")

for _, row in fixed.iterrows():
    print(f"  {row['Model']}: Accuracy={row['Accuracy']:.4f}, "
          f"F1={row['F1']:.4f}, ROC-AUC={row['ROC_AUC']:.4f}, PR-AUC={row['PR_AUC']:.4f}")
