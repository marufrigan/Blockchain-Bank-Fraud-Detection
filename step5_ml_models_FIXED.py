import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

TEMPLATE = "plotly_dark"

print("=== FIXED ML PIPELINE (Leakage-Free) ===")
print("Fix 1: Dropping label-derived account columns")
print("Fix 2: Using GroupShuffleSplit by account_id")

# Load cleaned data
df = pd.read_csv("output/cleaned_transactions.csv", index_col="original_index")
ids = pd.read_csv("output/transaction_ids.csv", index_col="original_index")
df["account_id"] = ids["account_id"]

print(f"\nOriginal shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

# ============================================================
# FIX 1: DROP ALL LABEL-DERIVED / LEAKY COLUMNS
# ============================================================
leaky_cols = [
    "fraud_rate", "risk_score", "is_high_risk", "is_fraudster",
    "fraud_count", "fraud_amount", "total_fraud_amount",
    "avg_fraud_amount", "fraud_frequency", "historical_fraud_rate",
    "account_fraud_rate", "account_risk_score"
]
dropped = [c for c in leaky_cols if c in df.columns]
if dropped:
    df.drop(columns=dropped, inplace=True)
    print(f"\nDropped leaky columns: {dropped}")
else:
    print("\nNo explicitly named leaky columns found — checking for suspicious columns...")
    # Auto-detect columns with 'fraud' or 'risk' in name (except is_fraud target)
    suspicious = [c for c in df.columns 
                  if ("fraud" in c.lower() or "risk" in c.lower()) 
                  and c != "is_fraud" and c != "account_id"]
    if suspicious:
        print(f"Auto-detected and dropping suspicious columns: {suspicious}")
        df.drop(columns=suspicious, inplace=True)
    else:
        print("No suspicious columns found. Proceeding with all features.")

print(f"Shape after dropping leaky columns: {df.shape}")

# ============================================================
# FIX 2: GROUP SPLIT BY ACCOUNT_ID (no account in both sets)
# ============================================================
groups = df["account_id"]
X = df.drop(columns=["is_fraud", "account_id"])
y = df["is_fraud"]

print(f"\nFeature count: {X.shape[1]}")
print(f"Unique accounts: {groups.nunique()}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"Test fraud rate:  {y_test.mean()*100:.2f}%")

# Verify no account overlap
train_accounts = set(groups.iloc[train_idx])
test_accounts  = set(groups.iloc[test_idx])
overlap = train_accounts & test_accounts
print(f"Account overlap between train/test: {len(overlap)} (should be 0)")

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

results = {}

def evaluate_model(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob)
    pr   = average_precision_score(y_true, y_prob)
    print(f"\n--- {name} Results (FIXED) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")
    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC_AUC": roc, "PR_AUC": pr
    }
    return acc, prec, rec, f1, roc, pr

def save_confusion_matrix(y_true, y_pred, name, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        colorscale="Blues", showscale=True)
    fig.update_layout(title=f"Confusion Matrix — {name} (Fixed)", template=TEMPLATE)
    fig.write_image(f"output/{filename}", width=800, height=600, scale=2)
    print(f"Saved {filename}")

def save_roc_curve(y_true, y_prob, name, filename):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"AUC = {auc_val:.4f}",
                             line=dict(color="#EF553B", width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="Random Classifier"))
    fig.update_layout(title=f"ROC Curve — {name} (Fixed)",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      template=TEMPLATE)
    fig.write_image(f"output/{filename}", width=900, height=700, scale=2)
    print(f"Saved {filename}")

def save_pr_curve(y_true, y_prob, name, filename):
    from sklearn.metrics import precision_recall_curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec_vals, y=prec_vals, mode="lines",
                             name=f"PR-AUC = {pr_auc:.4f}",
                             line=dict(color="#00CC96", width=3)))
    fig.update_layout(title=f"Precision-Recall Curve — {name} (Fixed)",
                      xaxis_title="Recall",
                      yaxis_title="Precision",
                      template=TEMPLATE)
    fig.write_image(f"output/{filename}", width=900, height=700, scale=2)
    print(f"Saved {filename}")

# ============================================================
# MODEL 1: RANDOM FOREST
# ============================================================
print("\n=== Training Random Forest (Fixed) ===")
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=42, n_jobs=-1, max_depth=15)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
save_confusion_matrix(y_test, rf_pred, "Random Forest", "fixed_rf_confusion_matrix.png")
save_roc_curve(y_test, rf_prob, "Random Forest", "fixed_rf_roc_curve.png")
save_pr_curve(y_test, rf_prob, "Random Forest", "fixed_rf_pr_curve.png")

fi_rf = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
fi_rf = fi_rf.sort_values("importance", ascending=False).head(15)
fi_rf.to_csv("output/fixed_rf_feature_importances.csv", index=False)
fig = px.bar(fi_rf, x="importance", y="feature", orientation="h",
             title="Random Forest — Top 15 Features (Fixed, Leakage-Free)",
             color="importance", color_continuous_scale="Blues", template=TEMPLATE)
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.write_image("output/fixed_rf_feature_importance.png", width=1000, height=700, scale=2)

pd.DataFrame({"y_true": y_test.values, "rf_fraud_score": rf_prob}).to_csv(
    "output/fixed_rf_predictions.csv", index=True)
print("Random Forest (Fixed) complete!")

# ============================================================
# MODEL 2: XGBOOST
# ============================================================
print("\n=== Training XGBoost (Fixed) ===")
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric="aucpr", verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)
save_confusion_matrix(y_test, xgb_pred, "XGBoost", "fixed_xgb_confusion_matrix.png")
save_roc_curve(y_test, xgb_prob, "XGBoost", "fixed_xgb_roc_curve.png")
save_pr_curve(y_test, xgb_prob, "XGBoost", "fixed_xgb_pr_curve.png")

fi_xgb = pd.DataFrame({"feature": X.columns, "importance": xgb.feature_importances_})
fi_xgb = fi_xgb.sort_values("importance", ascending=False).head(15)
fig = px.bar(fi_xgb, x="importance", y="feature", orientation="h",
             title="XGBoost — Top 15 Features (Fixed, Leakage-Free)",
             color="importance", color_continuous_scale="Oranges", template=TEMPLATE)
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.write_image("output/fixed_xgb_feature_importance.png", width=1000, height=700, scale=2)

pd.DataFrame({"y_true": y_test.values, "xgb_fraud_score": xgb_prob}).to_csv(
    "output/fixed_xgb_predictions.csv", index=True)
print("XGBoost (Fixed) complete!")

# ============================================================
# MODEL 3: GRAPH-ENHANCED LGBM
# ============================================================
print("\n=== Training Graph-Enhanced LightGBM (Fixed) ===")
graph_features = pd.read_csv("output/graph_features.csv")
graph_features["account_id"] = graph_features["account_id"].astype(str)

df_graph = df.copy() if "account_id" in df.columns else df.copy()
# Re-attach account_id for merge
df_graph["account_id"] = ids["account_id"].astype(str)
df_graph = df_graph.merge(graph_features, on="account_id", how="left")
df_graph[["degree_centrality", "clustering_coefficient", "is_ring_member"]] = \
    df_graph[["degree_centrality", "clustering_coefficient", "is_ring_member"]].fillna(0)

# Re-drop leaky columns and account_id
drop_again = [c for c in df_graph.columns 
              if ("fraud" in c.lower() or "risk" in c.lower()) 
              and c != "is_fraud"]
drop_again += ["account_id"]
df_graph.drop(columns=[c for c in drop_again if c in df_graph.columns], inplace=True)

X2 = df_graph.drop(columns=["is_fraud"])
y2 = df_graph["is_fraud"]
groups2 = ids["account_id"].iloc[:len(df_graph)]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx2, test_idx2 = next(gss2.split(X2, y2, groups=groups2))
X2_train = X2.iloc[train_idx2]
X2_test  = X2.iloc[test_idx2]
y2_train = y2.iloc[train_idx2]
y2_test  = y2.iloc[test_idx2]

lgbm = LGBMClassifier(
    n_estimators=300, class_weight="balanced",
    learning_rate=0.05, random_state=42, verbosity=-1)
lgbm.fit(X2_train, y2_train)
lgbm_pred = lgbm.predict(X2_test)
lgbm_prob = lgbm.predict_proba(X2_test)[:, 1]
evaluate_model("Graph-Enhanced LGBM", y2_test, lgbm_pred, lgbm_prob)
save_confusion_matrix(y2_test, lgbm_pred, "Graph-Enhanced LGBM", "fixed_lgbm_confusion_matrix.png")
save_roc_curve(y2_test, lgbm_prob, "Graph-Enhanced LGBM", "fixed_lgbm_roc_curve.png")
save_pr_curve(y2_test, lgbm_prob, "Graph-Enhanced LGBM", "fixed_lgbm_pr_curve.png")

pd.DataFrame({"y_true": y2_test.values, "lgbm_fraud_score": lgbm_prob}).to_csv(
    "output/fixed_lgbm_predictions.csv", index=True)
print("Graph-Enhanced LGBM (Fixed) complete!")

# ============================================================
# MODEL COMPARISON — BEFORE vs AFTER FIX
# ============================================================
print("\n=== Generating Fixed Model Comparison Chart ===")
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
results_df.to_csv("output/fixed_model_results_summary.csv", index=False)
print("\nFixed Model Results:")
print(results_df.to_string(index=False))

# Load original results for comparison
try:
    orig = pd.read_csv("output/model_results_summary.csv")
    orig["Version"] = "Before Fix (Inflated)"
    results_df["Version"] = "After Fix (Honest)"
    combined = pd.concat([orig, results_df], ignore_index=True)
    
    fig = px.bar(combined, x="Model", y="F1", color="Version", barmode="group",
                 title="F1 Score: Before vs After Leakage Fix",
                 color_discrete_map={
                     "Before Fix (Inflated)": "#EF553B",
                     "After Fix (Honest)": "#00CC96"
                 },
                 template=TEMPLATE)
    fig.update_layout(yaxis_title="F1 Score", yaxis_range=[0, 1.05])
    fig.write_image("output/fixed_before_vs_after_f1.png", width=1200, height=700, scale=2)
    print("Saved before vs after comparison chart")
    combined.to_csv("output/comparison_before_after_fix.csv", index=False)
except Exception as e:
    print(f"Could not load original results for comparison: {e}")

# Grouped bar chart for fixed results
metrics = ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
fig = go.Figure()
colors = ["#636EFA", "#EF553B", "#00CC96"]
for i, row in results_df.iterrows():
    fig.add_trace(go.Bar(
        name=row["Model"],
        x=metrics,
        y=[row[m] for m in metrics],
        marker_color=colors[i % len(colors)]
    ))
fig.update_layout(
    barmode="group",
    title="Fixed Model Performance Comparison (Leakage-Free, Group Split)",
    yaxis_title="Score",
    yaxis_range=[0, 1.05],
    template=TEMPLATE,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
fig.write_image("output/fixed_model_comparison.png", width=1400, height=700, scale=2)
print("Saved fixed_model_comparison.png")

# ============================================================
# PREPARE FIXED BLOCKCHAIN INPUT
# ============================================================
print("\n=== Preparing Fixed Blockchain Input ===")
xgb_preds = pd.read_csv("output/fixed_xgb_predictions.csv", index_col=0)
test_ids = ids.iloc[test_idx]

merged = xgb_preds.copy()
merged["transaction_id"] = test_ids["transaction_id"].values[:len(merged)]
merged["account_id"]     = test_ids["account_id"].values[:len(merged)]

fraud_sample = merged[merged["y_true"] == 1].sample(
    min(250, int(merged["y_true"].sum())), random_state=42)
legit_sample = merged[merged["y_true"] == 0].sample(250, random_state=42)
blockchain_input = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)
blockchain_input = blockchain_input[["transaction_id", "account_id", "xgb_fraud_score", "y_true"]]
blockchain_input.to_csv("output/blockchain_input.csv", index=False)

print(f"Fixed blockchain input: {len(blockchain_input)} transactions")
print(f"Fraud: {int(blockchain_input['y_true'].sum())}")
print(f"Legit: {int((blockchain_input['y_true']==0).sum())}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FIXED PIPELINE COMPLETE — PAPER-READY RESULTS")
print("="*60)
print(f"Leaky columns removed: {dropped if dropped else 'auto-detected'}")
print(f"Split method: GroupShuffleSplit by account_id")
print(f"Account overlap in train/test: {len(overlap)} (0 = correct)")
print("\nModel Results (Honest, Leakage-Free):")
for _, row in results_df.iterrows():
    print(f"  {row['Model']}: F1={row['F1']:.4f} | ROC-AUC={row['ROC_AUC']:.4f} | PR-AUC={row['PR_AUC']:.4f}")
print("\nAll output files:")
for f in sorted(os.listdir("output")):
    print(f"  output/{f}")
print("="*60)
