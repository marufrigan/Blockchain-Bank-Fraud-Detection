import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": True,
                     "legend.edgecolor": "black", "legend.fancybox": False, "mathtext.fontset": "stix"})
R = "results"; F = "figures"

# ---------- Fig. 2: PR curves + leakage audit ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.9), dpi=600, gridspec_kw={"width_ratios": [1.15, 1]})
styles = {"LightGBM": ("-", 1.1), "XGBoost": ((0, (4, 1.5)), 1.0), "RandomForest": ((0, (1, 1)), 1.0), "LogReg": ((0, (5, 1.5, 1, 1.5)), 0.9)}
labels = {"LightGBM": "LightGBM", "XGBoost": "XGBoost", "RandomForest": "RF", "LogReg": "LR"}
for m, (ls, lw) in styles.items():
    r = json.load(open(f"{R}/{m}_C2_ours.json"))
    a1.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"], ls=ls, lw=lw, color="black",
            label=f"{labels[m]} ({r['pr_auc'][0]:.3f})")
a1.axhline(0.017143, color="black", lw=0.5, ls=":")
a1.text(0.50, 0.035, "prevalence 1.71%", fontsize=5.8)
a1.set_xlabel("Recall"); a1.set_ylabel("Precision"); a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
a1.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(0.0, 0.08), handlelength=2.6, borderpad=0.3, labelspacing=0.25)
a1.set_title("(a) PR curves, C2 (PR-AUC)", fontsize=7.5)
cfgs = [("C4_tx_only", "C4"), ("C3_ours_nograph", "C3"), ("C2_ours", "C2"), ("C1_repo_fixed", "C1"), ("C0_naive", "C0")]
vals = [json.load(open(f"{R}/XGBoost_{c}.json"))["pr_auc"] for c, _ in cfgs]
hatches = ["", "", "", "////", "xxxx"]
for i, ((c, lab), v, h) in enumerate(zip(cfgs, vals, hatches)):
    a2.bar(i, v[0], yerr=v[1], color="white" if h else ("#bfbfbf" if c != "C2_ours" else "#606060"), edgecolor="black",
           hatch=h, lw=0.6, width=0.7, capsize=1.5, error_kw={"lw": 0.6})
    a2.text(i, v[0] + 0.015, f"{v[0]:.3f}", ha="center", fontsize=5.6)
a2.set_xticks(range(len(cfgs))); a2.set_xticklabels([l for _, l in cfgs])
a2.set_ylim(0, 1.22); a2.set_ylabel("PR-AUC (XGBoost)")
a2.axvline(2.5, color="black", lw=0.5, ls=(0, (3, 2)))
a2.text(1.0, 1.13, "leakage-free", ha="center", fontsize=6); a2.text(3.5, 1.13, "leaky", ha="center", fontsize=6)
a2.set_title("(b) Leakage audit", fontsize=7.5)
for a in (a1, a2): a.tick_params(labelsize=6.5)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f"{F}/fig2_pr_audit.png", dpi=600)

# ---------- Fig. 3: gas and batching ----------
b = json.load(open(f"{R}/bench.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=600, gridspec_kw={"width_ratios": [1.1, 1]})
names = ["APPROVE", "REVIEW", "FLAG", "REJECT"]
v1 = [b["v1_gas_by_decision"].get(str(k), {}).get("mean", np.nan) / 1000 for k in range(4)]
v2 = [b["v2_gas_by_decision"].get(str(k), {}).get("mean", np.nan) / 1000 for k in range(4)]
x = np.arange(4); w = 0.38
a1.bar(x - w / 2, v1, w, color="white", edgecolor="black", hatch="////", lw=0.6, label="V1 (repository)")
a1.bar(x + w / 2, v2, w, color="#808080", edgecolor="black", lw=0.6, label="V2 (hardened)")
a1.set_xticks(x); a1.set_xticklabels(["APPR.", "REV.", "FLAG", "REJ."], fontsize=6.3)
a1.set_ylabel("Gas per call (k)"); a1.set_ylim(0, max(v1) * 1.55)
a1.legend(fontsize=5.4, loc="upper left", borderpad=0.3, ncol=2, columnspacing=0.6, handlelength=1.4)
a1.set_title("(a) Gas by decision", fontsize=7.5)
Bs = [d["B"] for d in b["batch"]]; g = [d["gas_per_tx"] / 1000 for d in b["batch"]]; tp = [d["tps"] for d in b["batch"]]
a2.plot(range(len(Bs)), g, "-o", color="black", ms=2.8, lw=0.9, mfc="white", label="gas / tx")
a2.set_xticks(range(len(Bs))); a2.set_xticklabels(Bs); a2.set_xlabel("Batch size B")
a2.set_ylabel("Gas per tx (k)"); a2.set_ylim(0, max(g) * 1.2)
a3 = a2.twinx(); a3.plot(range(len(Bs)), tp, "--s", color="black", ms=2.6, lw=0.9, mfc="black", label="tx/s")
a3.set_ylabel("Throughput (tx/s)"); a3.set_ylim(0, max(tp) * 1.2); a3.tick_params(labelsize=6.5)
h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels()
a2.legend(h1 + h2, l1 + l2, fontsize=5.6, loc="lower right", borderpad=0.3)
a2.set_title("(b) V2 batching", fontsize=7.5)
for a in (a1, a2): a.tick_params(labelsize=6.5)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(f"{F}/fig3_gas_batch.png", dpi=600)
print("figs ok")
