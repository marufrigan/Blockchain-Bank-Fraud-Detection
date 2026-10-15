import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Circle, Polygon, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "mathtext.fontset": "stix"})
fig = plt.figure(figsize=(7.16, 3.9), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 143.2); ax.set_ylim(0, 78); ax.axis("off")

K = "black"
def box(x, y, w, h, title, sub=None, hatch=None, bold=True, fs=6.6, sfs=5.6, fc="white", lw=0.7):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=K, lw=lw, hatch=hatch, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sfs, style="italic", zorder=4, linespacing=1.05)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", zorder=4, linespacing=1.05)
    return (x, y, w, h)

def arr(x1, y1, x2, y2, dashed=False, lw=0.7, cs="arc3,rad=0", head=4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=f"-|>,head_length={head*0.12},head_width={head*0.06}",
                        connectionstyle=cs, lw=lw, color=K, linestyle=(0, (3, 2)) if dashed else "-", zorder=5,
                        shrinkA=0, shrinkB=0, mutation_scale=10)
    ax.add_patch(a)

def poly(pts, dashed=False, lw=0.7):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=K, lw=lw, ls=(0, (3, 2)) if dashed else "-", zorder=5, solid_capstyle="butt")
    arr(xs[-2], ys[-2], xs[-1], ys[-1], dashed=dashed, lw=lw)

def lane(y, h, label, shade):
    ax.add_patch(Rectangle((1, y), 113, h, fc=shade, ec=K, lw=0.6, zorder=0))
    ax.add_patch(Rectangle((1, y), 5, h, fc="white", ec=K, lw=0.6, zorder=1))
    ax.text(3.5, y + h / 2, label, rotation=90, ha="center", va="center", fontsize=6.2, weight="bold", zorder=2)

# ---------- icons ----------
def db_icon(cx, cy, s=1.0):
    w, h = 3.4 * s, 4.0 * s
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec=K, lw=0.6, zorder=6))
    for yy in (cy - h / 2, cy + h / 2):
        ax.add_patch(Ellipse((cx, yy), w, 1.3 * s, fc="white", ec=K, lw=0.6, zorder=6))
    ax.add_patch(Ellipse((cx, cy + h / 2), w, 1.3 * s, fc="white", ec=K, lw=0.6, zorder=7))
    ax.plot([cx - w / 2, cx + w / 2], [cy + h / 2 - 0.0, cy + h / 2], color="white", lw=0, zorder=7)

def graph_icon(cx, cy):
    pts = [(cx - 1.8, cy + 1.2), (cx + 1.6, cy + 1.6), (cx, cy - 0.2), (cx - 1.4, cy - 1.7), (cx + 1.8, cy - 1.4)]
    for i, j in [(0, 2), (1, 2), (2, 3), (2, 4), (0, 1), (3, 4)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=K, lw=0.5, zorder=6)
    for k, p in enumerate(pts):
        ax.add_patch(Circle(p, 0.55, fc="black" if k in (0, 2) else "white", ec=K, lw=0.5, zorder=7))

def lock_icon(cx, cy):
    ax.add_patch(Rectangle((cx - 1.4, cy - 1.6), 2.8, 2.2, fc="black", ec=K, lw=0.5, zorder=7))
    t = __import__("numpy").linspace(0, 3.1416, 30)
    ax.plot(cx + 0.95 * __import__("numpy").cos(t), cy + 0.6 + 1.0 * __import__("numpy").sin(t), color=K, lw=0.8, zorder=7)
    ax.add_patch(Circle((cx, cy - 0.5), 0.3, fc="white", ec="white", zorder=8))

def chain_icon(cx, cy):
    for dx in (-1.1, 1.1):
        ax.add_patch(FancyBboxPatch((cx + dx - 1.3, cy - 0.7), 2.6, 1.4, boxstyle="round,pad=0,rounding_size=0.7", fc="none", ec=K, lw=0.8, zorder=7))

def person_icon(cx, cy):
    ax.add_patch(Circle((cx, cy + 1.2), 0.85, fc="black", zorder=7))
    ax.add_patch(Polygon([(cx - 1.5, cy - 1.6), (cx + 1.5, cy - 1.6), (cx + 1.1, cy + 0.1), (cx - 1.1, cy + 0.1)], closed=True, fc="black", zorder=7))

def doc_icon(cx, cy):
    ax.add_patch(Polygon([(cx - 1.3, cy - 1.7), (cx + 1.3, cy - 1.7), (cx + 1.3, cy + 1.0), (cx + 0.6, cy + 1.7), (cx - 1.3, cy + 1.7)], closed=True, fc="white", ec=K, lw=0.6, zorder=7))
    for yy in (cy + 0.6, cy - 0.1, cy - 0.8):
        ax.plot([cx - 0.8, cx + 0.8], [yy, yy], color=K, lw=0.4, zorder=8)

# ---------- lanes ----------
lane(58, 19, "Data & features", "#f2f2f2")
lane(39, 18, "Detection", "#e6e6e6")
lane(20.5, 17.5, "Scoring oracle", "#f2f2f2")
lane(1, 18.5, "Decision ledger", "#e6e6e6")

# governance column
ax.add_patch(Rectangle((116, 1), 26.2, 76, fc="white", ec=K, lw=0.6, zorder=0))
ax.add_patch(Rectangle((116, 71), 26.2, 6, fc="#d9d9d9", ec=K, lw=0.6, zorder=1))
ax.text(129.1, 74, "Governance & feedback", ha="center", va="center", fontsize=6.4, weight="bold", zorder=2)

# ---------- Lane A ----------
A1 = box(9, 71.4, 28, 4.9, "Transaction stream", None, fs=6.4); ax.texts[-1].set_position((24.5, 74.4))
ax.text(24.5, 72.4, "1.0 M records, 2022 to 2024", ha="center", va="center", fontsize=5.4, style="italic", zorder=4)
A2 = box(9, 65.4, 28, 4.9, "Account KYC profile", None, fs=6.4); ax.texts[-1].set_position((24.5, 68.4))
ax.text(24.5, 66.4, "49,903 accounts, static fields", ha="center", va="center", fontsize=5.4, style="italic", zorder=4)
A3 = box(9, 59.4, 28, 4.9, "Shared-attribute graph", None, fs=6.4); ax.texts[-1].set_position((24.5, 62.4))
ax.text(24.5, 60.4, "6,783 nodes, 7,411 edges", ha="center", va="center", fontsize=5.4, style="italic", zorder=4)
db_icon(12.3, 73.85, 0.55); db_icon(12.3, 67.85, 0.55); graph_icon(12.3, 61.85)
B_ = [("Point-in-time features", r"$n_{<t}$, $c_{24h}$, $c_{7d}$, amount$/\bar{a}_{<t}$, new country/device", 73.85),
      ("Static profile features", "age, credit limit, type, 2FA, declared activity", 67.85),
      ("Unsupervised graph features", "degree, weighted degree, k-core, component size", 61.85)]
for title, sub, yc in B_:
    box(43, yc - 2.45, 40, 4.9, title, None, fs=6.4); ax.texts[-1].set_position((63, yc + 0.75))
    ax.text(63, yc - 1.15, sub, ha="center", va="center", fontsize=5.3, style="italic", zorder=4)
    arr(37, yc, 43, yc); arr(83, yc, 89, yc)
A7 = box(89, 59.4, 22, 16.9, "Feature\nassembly", None, fs=6.8); ax.texts[-1].set_position((100, 69.6))
ax.text(100, 63.6, r"$\mathbf{x}_t\in\mathbb{R}^{d}$" "\n(labels and ring IDs\nnever read)", ha="center", va="center", fontsize=5.3, style="italic", zorder=4)

# ---------- Lane B ----------
B1 = box(9, 41.0, 22, 11.0, "Leakage guard", "drop label-derived and\nfull-history aggregates;\nexclude ring_id", sfs=5.3)
B2 = box(36, 41.0, 20, 11.0, "Account-disjoint\nCV", "StratifiedGroupKFold\n(k = 5)", fs=6.4, sfs=5.3)
B3 = box(61, 41.0, 22, 11.0, "Classifier", "LR / RF / XGBoost /\nLightGBM", sfs=5.3)
B4 = box(88, 41.0, 23, 11.0, "Risk policy", r"$s<\theta_r$: APPROVE" "\n" r"$\theta_r\leq s<\theta_f$: REVIEW" "\n" r"$s\geq\theta_f$: FLAG", sfs=5.2)
poly([(97, 59.4), (97, 56.4), (20, 56.4), (20, 52.0)])
arr(31, 46.5, 36, 46.5); arr(56, 46.5, 61, 46.5); arr(83, 46.5, 88, 46.5)
ax.text(85.5, 47.8, r"$s_t$", fontsize=6.3, ha="center", zorder=6)

# ---------- Lane C ----------
C1 = box(9, 23.5, 28, 10.5, "Pseudonymiser", r"$r=\mathrm{keccak256}(\sigma\,\|\,id)$" "\nno plaintext IDs on chain", sfs=5.3)
C2 = box(43, 23.5, 24, 10.5, "Batch builder", r"$B\in\{1,10,25,50,100\}$" "\ntransactions per call", sfs=5.3)
C3 = box(73, 23.5, 22, 10.5, "Oracle signer", "authorised account\n(isOracle = true)", sfs=5.3)
lock_icon(91.2, 31.1)
doc_icon(103.5, 29.6)
ax.text(103.5, 25.3, "model\nversion", ha="center", va="center", fontsize=5.0, zorder=8)
poly([(99.5, 41.0), (99.5, 36.6), (23, 36.6), (23, 34.0)])
ax.text(100.6, 37.5, r"$(id_t,\ acct_t,\ s_t)$", fontsize=5.6, ha="left", va="center", zorder=6)
arr(37, 28.75, 43, 28.75); arr(67, 28.75, 73, 28.75)

# ---------- Lane D ----------
D0 = Rectangle((8.2, 2.6), 104.6, 15.4, fc="none", ec=K, lw=0.9, ls=(0, (5, 2)), zorder=2)
ax.add_patch(D0)
ax.text(22, 16.2, "FraudDetectionV2 (EVM smart contract)", fontsize=6.0, weight="bold", ha="left", va="center", zorder=6)
chain_icon(107.5, 16.2)
def hbox(x, y, w, h, t):
    box(x, y, w, h, "", None, hatch="////", fc="white")
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.2, weight="bold", zorder=6,
            bbox=dict(fc="white", ec=K, lw=0.4, pad=1.2))
hbox(11, 4.5, 17, 9.0, "require\nisOracle")
hbox(32, 4.5, 17, 9.0, "require\nwrite-once")
D3 = box(53, 4.5, 17, 9.0, "frozen[acct]\n?", fs=6.2)
D4 = box(74, 4.5, 15, 9.0, "Decision", "A / R / F / REJECT", sfs=5.0)
D5 = box(93, 4.5, 18, 9.0, "Ledger +\nevents", fs=6.2)
poly([(84, 23.5), (84, 20.6), (19.5, 20.6), (19.5, 13.5)])
arr(28, 9, 32, 9); arr(49, 9, 53, 9); arr(70, 9, 74, 9); arr(89, 9, 93, 9)
ax.text(71.9, 10.3, "no", fontsize=5.2, ha="center", zorder=6)
ax.text(66.5, 15.0, "yes: REJECT", fontsize=5.2, ha="center", zorder=6)
poly([(61.5, 13.5), (61.5, 16.6), (81.5, 16.6), (81.5, 13.5)])
poly([(103.5, 24.0), (103.5, 21.8), (100.5, 21.8), (100.5, 13.5)], dashed=True)

# ---------- Governance column ----------
G1 = box(118, 53.0, 22, 14.0, "Analyst review\nqueue", "REVIEW and FLAG\ncases triaged", fs=6.2, sfs=5.2)
G2 = box(118, 25.0, 22, 18.0, "Compliance\nofficer (owner)", "unfreeze(acct, reason)\nsetThresholds(" r"$\theta_r,\theta_f$" ", v)", fs=6.2, sfs=5.0)
person_icon(129, 47.8)
G3 = box(118, 4.5, 22, 13.0, "Audit trail", "Decided / Frozen /\nUnfrozen events", fs=6.2, sfs=5.2)
arr(111, 9, 118, 9)
poly([(111, 11.8), (114.8, 11.8), (114.8, 62), (118, 62)])
poly([(118, 55.0), (116.2, 55.0), (116.2, 53.9), (72, 53.9), (72, 52.0)], dashed=True)
ax.text(93, 55.0, "confirmed labels (retraining)", fontsize=5.3, style="italic", ha="center", va="center", zorder=6)
poly([(118, 30), (112.0, 30), (112.0, 18.0)], dashed=True)
ax.text(113.0, 24.0, "governs", fontsize=5.0, style="italic", ha="center", va="center", rotation=90, zorder=6)

fig.savefig("figures/fig1_architecture.png", dpi=600)
fig.savefig("figures/fig1_architecture.pdf")
print("ok")
