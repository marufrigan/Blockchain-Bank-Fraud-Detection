"""Risk-policy calibration, freeze simulation and on-chain input preparation.
Usage: python3 policy.py <model_tag>   e.g. LightGBM_C2_ours"""
import sys, json
import numpy as np, pandas as pd

tag = sys.argv[1]
d = pd.read_parquet(f"results/oof_{tag}.parquet").sort_values("ts").reset_index(drop=True)
cal = d[d.fold.isin([0, 1, 2])]; ev = d[d.fold.isin([3, 4])].copy()

def pick(cal, prec_target=0.90, recall_target=0.90):
    s = np.sort(cal.score.to_numpy())[::-1]; y = cal.y.to_numpy()[np.argsort(-cal.score.to_numpy())]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1); rec = tp / y.sum()
    # theta_f: lowest score threshold whose FLAG band precision is still >= target
    ok = np.flatnonzero(prec >= prec_target); i_f = ok.max()
    # theta_r: score at which REVIEW+FLAG reaches the recall target
    i_r = np.argmax(rec >= recall_target)
    return round(float(s[i_r]), 4), round(float(s[i_f]), 4)

def bands(df, tr, tf):
    s, y = df.score.to_numpy(), df.y.to_numpy(); out = {}
    for name, m in [("APPROVE", s < tr), ("REVIEW", (s >= tr) & (s < tf)), ("FLAG", s >= tf)]:
        out[name] = {"n": int(m.sum()), "share": float(m.mean()), "fraud": int(y[m].sum()),
                     "precision": float(y[m].mean()) if m.sum() else 0.0}
    out["recall_flag"] = out["FLAG"]["fraud"] / y.sum()
    out["recall_review_or_flag"] = (out["FLAG"]["fraud"] + out["REVIEW"]["fraud"]) / y.sum()
    out["alert_rate"] = out["FLAG"]["share"] + out["REVIEW"]["share"]
    return out

def simulate_contract(df, tr, tf, hold_h=None):
    """Exact Python replica of the contract decision rule (freeze on FLAG, reject while frozen).
    hold_h=None: permanent freeze (original contract). Otherwise the account is released by the
    officer's unfreeze() hold_h hours after the FLAG (review SLA)."""
    frozen = {}; dec = []
    ts = df.ts.astype("datetime64[s]").astype("int64").to_numpy()
    for a, s, t in zip(df.account_id.to_numpy(), df.score.to_numpy(), ts):
        if a in frozen and (hold_h is None or t < frozen[a] + hold_h * 3600): dec.append(3)
        elif s >= tf: dec.append(2); frozen[a] = t
        elif s >= tr: dec.append(1)
        else: dec.append(0)
    return np.array(dec)

def onchain_summary(df, dec):
    y = df.y.to_numpy(); r = {}
    for k, n in enumerate(["APPROVE", "REVIEW", "FLAG", "REJECT"]):
        m = dec == k; r[n] = {"n": int(m.sum()), "fraud": int(y[m].sum()), "legit": int((m & (y == 0)).sum())}
    r["frozen_accounts"] = int(df.account_id[dec == 2].nunique())
    fa = set(df.account_id[dec == 2]); fraudster = set(df.account_id[df.y == 1])
    r["frozen_accounts_with_any_fraud"] = len(fa & fraudster)
    r["fraud_blocked"] = r["FLAG"]["fraud"] + r["REJECT"]["fraud"]
    r["fraud_blocked_rate"] = r["fraud_blocked"] / y.sum()
    r["legit_blocked"] = r["FLAG"]["legit"] + r["REJECT"]["legit"]
    r["legit_blocked_rate"] = r["legit_blocked"] / (y == 0).sum()
    return r

res = {"model": tag, "n_cal": len(cal), "n_eval": len(ev), "eval_fraud": int(ev.y.sum())}
tr, tf = pick(cal); res["ours"] = {"theta_r": tr, "theta_f": tf, "bands": bands(ev, tr, tf)}
res["repo"] = {"theta_r": 0.60, "theta_f": 0.85, "bands": bands(ev, 0.60, 0.85)}
for key in ("ours", "repo"):
    dec = simulate_contract(ev, res[key]["theta_r"], res[key]["theta_f"])
    res[key]["contract_sim"] = onchain_summary(ev, dec)
    for h in (24, 72):
        res[key][f"contract_sim_hold{h}h"] = onchain_summary(ev, simulate_contract(ev, res[key]["theta_r"], res[key]["theta_f"], h))
json.dump(res, open(f"results/policy_{tag}.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# on-chain stream: first 2,000 chronologically ordered evaluation transactions (natural prevalence)
w = ev.iloc[:2000].copy()
w["score"] = w.score.clip(0, 1).round(4)
w[["transaction_id", "account_id", "score", "y"]].to_csv(f"results/onchain_input_{tag}.csv", index=False)
for key in ("ours", "repo"):
    np.save(f"results/onchain_expected_{key}_{tag}.npy", simulate_contract(w, res[key]["theta_r"], res[key]["theta_f"]))
print("window fraud:", int(w.y.sum()), "span:", w.ts.min(), w.ts.max())
