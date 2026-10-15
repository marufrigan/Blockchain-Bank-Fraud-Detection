"""Leakage-controlled fraud detection experiments for the paper.
Outputs results/*.json and OOF score files. Run: python3 ml.py <stage>
stage = features | audit | models | all
"""
import sys, json, time, os
import numpy as np, pandas as pd, networkx as nx
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA = os.environ.get("DATA_DIR", "..")   # repository root holds the CSV files
D = "work"; R = "results"; os.makedirs(R, exist_ok=True); os.makedirs(D, exist_ok=True)
SEED = 42; K = 5

TX = ["hour_of_day", "day_of_week", "is_weekend", "amount", "merchant_category", "mcc_code", "merchant_country",
      "card_present", "device_type", "device_known", "ip_risk_score", "is_foreign_txn", "time_since_last_s",
      "velocity_1h", "amount_vs_avg_ratio", "account_age_days", "has_2fa", "credit_limit"]
CATS = ["merchant_category", "merchant_country", "device_type", "account_type"]
STATIC = ["account_type", "avg_txn_amount", "avg_monthly_txns"]
CAUSAL = ["n_prior", "cnt_24h", "cnt_7d", "prior_mean_amt", "amt_ratio_prior", "new_country", "new_device", "prior_foreign_rate"]
GRAPH = ["g_degree", "g_wdegree", "g_kcore", "g_comp_size", "g_n_types", "g_max_conn"]
AGG_FULL = ["total_transactions", "total_amount", "avg_amount", "max_amount", "pct_foreign", "avg_velocity",
            "unique_countries", "unique_categories", "avg_ip_risk"]
LABEL_DERIVED = ["fraud_count", "fraud_amount", "fraud_rate", "is_fraudster", "risk_score", "is_high_risk"]
REPO_GRAPH = ["degree_centrality", "clustering_coefficient", "is_ring_member"]

CONFIGS = {
    "C0_naive":        TX + STATIC + AGG_FULL + LABEL_DERIVED + REPO_GRAPH,
    "C1_repo_fixed":   TX + STATIC + AGG_FULL + REPO_GRAPH,
    "C2_ours":         TX + STATIC + CAUSAL + GRAPH,
    "C3_ours_nograph": TX + STATIC + CAUSAL,
    "C4_tx_only":      TX,
}

def build_features():
    tx = f"{DATA}/transactions.csv"
    t = pd.read_csv(tx if os.path.exists(tx) else tx + ".gz")
    a = pd.read_csv(f"{DATA}/account_profiles.csv")
    e = pd.read_csv(f"{DATA}/network_edges.csv")
    t["ts"] = pd.to_datetime(t["timestamp"])
    t = t.sort_values(["account_id", "ts", "transaction_id"]).reset_index(drop=True)
    g = t.groupby("account_id", sort=False)
    # ---- causal (point-in-time) behavioural features: only strictly earlier transactions ----
    t["n_prior"] = g.cumcount()
    csum = g["amount"].cumsum() - t["amount"]
    t["prior_mean_amt"] = (csum / t["n_prior"].replace(0, np.nan)).fillna(-1)
    t["amt_ratio_prior"] = np.where(t["n_prior"] > 0, t["amount"] / t["prior_mean_amt"].where(t["prior_mean_amt"] > 0, np.nan), -1)
    t["amt_ratio_prior"] = t["amt_ratio_prior"].fillna(-1)
    fsum = g["is_foreign_txn"].cumsum() - t["is_foreign_txn"]
    t["prior_foreign_rate"] = (fsum / t["n_prior"].replace(0, np.nan)).fillna(-1)
    t["new_country"] = (~t.duplicated(["account_id", "merchant_country"])).astype(int)
    t["new_device"] = (~t.duplicated(["account_id", "device_type"])).astype(int)
    ts_ns = t["ts"].astype("datetime64[ns]").astype("int64").to_numpy(); acc = t["account_id"].to_numpy()
    cnt24 = np.zeros(len(t), dtype=np.int32); cnt7 = np.zeros(len(t), dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(acc[1:] != acc[:-1]) + 1, len(t)]
    for s, en in zip(starts[:-1], starts[1:]):
        x = ts_ns[s:en]; idx = np.arange(en - s)
        cnt24[s:en] = idx - np.searchsorted(x, x - 86400 * 10**9, side="left")
        cnt7[s:en] = idx - np.searchsorted(x, x - 7 * 86400 * 10**9, side="left")
    t["cnt_24h"] = cnt24; t["cnt_7d"] = cnt7
    # ---- unsupervised structural graph features (ring_id / both_fraud never read) ----
    G = nx.Graph()
    for r in e[["account_a", "account_b", "shared_type", "connection_count"]].itertuples(index=False):
        if G.has_edge(r[0], r[1]):
            G[r[0]][r[1]]["w"] += r[3]; G[r[0]][r[1]]["types"].add(r[2])
        else:
            G.add_edge(r[0], r[1], w=r[3], types={r[2]})
    core = nx.core_number(G); comp = {}
    for c in nx.connected_components(G):
        for n in c: comp[n] = len(c)
    gf = pd.DataFrame({"account_id": list(G.nodes())})
    gf["g_degree"] = [G.degree(n) for n in gf.account_id]
    gf["g_wdegree"] = [G.degree(n, weight="w") for n in gf.account_id]
    gf["g_kcore"] = [core[n] for n in gf.account_id]
    gf["g_comp_size"] = [comp[n] for n in gf.account_id]
    gf["g_n_types"] = [len(set().union(*[G[n][m]["types"] for m in G[n]])) for n in gf.account_id]
    gf["g_max_conn"] = [max(G[n][m]["w"] for m in G[n]) for n in gf.account_id]
    # ---- repository-style graph features (for the leakage audit only) ----
    H = nx.Graph(); H.add_edges_from(zip(e.account_a, e.account_b))
    dc = nx.degree_centrality(H); cc = nx.clustering(H)
    ring = set(e.loc[e.ring_id.notna(), "account_a"]) | set(e.loc[e.ring_id.notna(), "account_b"])
    rg = pd.DataFrame({"account_id": list(H.nodes())})
    rg["degree_centrality"] = rg.account_id.map(dc); rg["clustering_coefficient"] = rg.account_id.map(cc)
    rg["is_ring_member"] = rg.account_id.isin(ring).astype(int)
    t = t.drop(columns=["account_age_days", "has_2fa", "credit_limit"]).merge(
        a, on="account_id", how="left").merge(gf, on="account_id", how="left").merge(rg, on="account_id", how="left")
    t[GRAPH + REPO_GRAPH] = t[GRAPH + REPO_GRAPH].fillna(0)
    for c in CATS:
        t[c] = t[c].astype("category").cat.codes
    t = t.sort_values("ts").reset_index(drop=True)
    t.to_parquet(f"{D}/features.parquet")
    stats = {"n_tx": len(t), "n_acc": int(t.account_id.nunique()), "fraud_rate": float(t.is_fraud.mean()),
             "n_fraud": int(t.is_fraud.sum()), "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
             "n_components": nx.number_connected_components(G), "ts_min": str(t.ts.min()), "ts_max": str(t.ts.max())}
    json.dump(stats, open(f"{R}/data_stats.json", "w"), indent=1); print(stats)

def make_model(name, n_pos, n_neg):
    spw = n_neg / max(n_pos, 1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="aucpr", verbosity=0)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=2, random_state=SEED, verbosity=-1)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=5, max_samples=0.3,
                                      class_weight="balanced_subsample", n_jobs=2, random_state=SEED)
    if name == "LogReg":
        return None

def fit_predict(name, Xtr, ytr, Xte, cols):
    if name == "LogReg":
        cats = [c for c in cols if c in CATS]; nums = [c for c in cols if c not in CATS]
        pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cats), ("sc", StandardScaler(), nums)])
        m = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1], m
    m = make_model(name, int(ytr.sum()), int((ytr == 0).sum())); m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m

def rec_at_fpr(y, p, f=0.01):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(f, fpr, tpr))

def run(name, cfg, save_oof=False):
    df = pd.read_parquet(f"{D}/features.parquet")
    cols = CONFIGS[cfg]; X = df[cols]; y = df["is_fraud"].to_numpy(); grp = df["account_id"].to_numpy()
    skf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df)); fold_of = np.zeros(len(df), dtype=int); per = []; t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(X, y, grp)):
        # inner account-disjoint validation split for threshold selection
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + k)
        itr, iva = next(gss.split(tr, y[tr], grp[tr])); itr, iva = tr[itr], tr[iva]
        pv, m = fit_predict(name, X.iloc[itr], y[itr], X.iloc[iva], cols)
        pr, rc, th = precision_recall_curve(y[iva], pv); f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        thr = float(th[np.argmax(f[:-1])])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p; fold_of[te] = k
        pred = (p >= thr).astype(int)
        per.append({"fold": k, "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p),
                    "f1": f1_score(y[te], pred), "precision": float((pred & y[te]).sum() / max(pred.sum(), 1)),
                    "recall": float((pred & y[te]).sum() / y[te].sum()), "rec_at_1fpr": rec_at_fpr(y[te], p),
                    "thr": thr, "n_test": int(len(te)), "test_fraud": int(y[te].sum())})
        print(name, cfg, per[-1], f"{time.time()-t0:.0f}s", flush=True)
        if k == 0 and name in ("XGBoost", "LightGBM") and cfg == "C2_ours":
            imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
            imp.to_csv(f"{R}/importance_{name}.csv")
    res = {"model": name, "config": cfg, "folds": per, "time_s": time.time() - t0}
    for kk in ["pr_auc", "roc_auc", "f1", "precision", "recall", "rec_at_1fpr"]:
        v = [d[kk] for d in per]; res[kk] = [float(np.mean(v)), float(np.std(v))]
    # pooled PR curve (downsampled) for plotting
    pr, rc, _ = precision_recall_curve(y, oof); idx = np.linspace(0, len(pr) - 1, 400).astype(int)
    res["pr_curve"] = {"precision": pr[idx].tolist(), "recall": rc[idx].tolist()}
    json.dump(res, open(f"{R}/{name}_{cfg}.json", "w"))
    if save_oof:
        pd.DataFrame({"transaction_id": df.transaction_id, "account_id": df.account_id, "ts": df.ts, "y": y,
                      "score": oof, "fold": fold_of}).to_parquet(f"{R}/oof_{name}_{cfg}.parquet")
    print("DONE", name, cfg, {k: res[k] for k in ["pr_auc", "roc_auc", "f1", "rec_at_1fpr"]}, flush=True)

if __name__ == "__main__":
    st = sys.argv[1]
    if st in ("features", "all"): build_features()
    if st in ("models", "all"):
        for m in ["XGBoost", "LightGBM", "LogReg", "RandomForest"]:
            run(m, "C2_ours", save_oof=m in ("XGBoost", "LightGBM"))
    if st in ("audit", "all"):
        for c in ["C4_tx_only", "C3_ours_nograph", "C1_repo_fixed", "C0_naive"]:
            run("XGBoost", c)
    if st == "one":
        run(sys.argv[2], sys.argv[3])
