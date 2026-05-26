"""
Grand multipanel Oxi-DIA validation figure
==========================================
Row 1  A  Calibration curve (mean ± SD)          B  Per-site scatter onto identity
Row 2  C  Residual scatter                        D  CV% by level (0% excluded)
Row 3  E  R²   F  RMSE/MAE/SD   G  Bias   H  MAPE%
Row 4  I  Rep 1 vs 2            J  Rep 1 vs 3     K  Rep 2 vs 3

Output: fig_grand.png (600 DPI) + fig_grand.pdf
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Global style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          7,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         600,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Palette ───────────────────────────────────────────────────────────────────
MS1_COL    = "#2166AC"
MS2_COL    = "#D6604D"
COMB_COL   = "#1A9641"
EXPECT_COL = "#252525"
GRID_COL   = "#E0E0E0"
BAND_COL   = "#F4F4F4"

PALETTE = {
    0:"#4E79A7", 20:"#59A14F", 40:"#F28E2B", 60:"#E15759",
    80:"#76B7B2", 90:"#B07AA1", 95:"#FF9DA7", 100:"#9C755F",
}
LEVEL_ORDER  = [0, 20, 40, 60, 80, 90, 95, 100]
ESTIMATORS   = ["MS1", "MS2", "Combined"]
EST_COLORS   = [MS1_COL, MS2_COL, COMB_COL]
EST_MARKERS  = ["o", "s", "^"]
DASHES       = {"MS1": "-", "MS2": (0,(4,2)), "Combined": (0,(2,2))}

# ── Data ──────────────────────────────────────────────────────────────────────
raw = {
    "Run": [
        "James_0_S22","James_0_S23","James_0_S24",
        "James_100_S1","James_100_S2","James_100_S3",
        "James_20_S19","James_20_S20","James_20_S21",
        "James_40_S16","James_40_S17","James_40_S18",
        "James_60_S13","James_60_S14","James_60_S15",
        "James_80_S10","James_80_S11","James_80_S12",
        "James_90_S7","James_90_S8","James_90_S9",
        "James_95_S4","James_95_S5","James_95_S6",
    ],
    "Total_C_Peptides": [
        9341,9884,9749,18560,18233,18119,10805,11302,11158,
        12064,12439,12737,14286,14777,15087,16291,17156,16623,
        17341,17262,16973,17865,17884,17780,
    ],
    "Light_C": [
        41,58,75,18077,17752,17611,1992,2256,2330,4246,4633,4924,
        7756,8288,8747,12374,12777,12645,15078,15155,14672,
        16352,16327,16111,
    ],
    "Heavy_C": [
        7708,8192,8020,472,472,499,7839,8016,7848,7209,7199,7201,
        6185,6136,5983,3772,4220,3830,2209,2049,2238,1474,1515,1636,
    ],
    "Unlabeled_C": [
        1592,1634,1654,11,9,9,974,1030,980,609,607,612,
        345,353,357,145,159,148,54,58,63,39,42,33,
    ],
}
df = pd.DataFrame(raw)
df["Target_Ox"] = df["Run"].str.extract(r"James_(\d+)_")[0].astype(int)
df["Total_C"]   = df["Light_C"] + df["Heavy_C"] + df["Unlabeled_C"]
df["MS1"]       = df["Light_C"] / df["Total_C"]
df["MS2"]       = df["Light_C"] / (df["Light_C"] + df["Heavy_C"])
df["Combined"]  = (df["MS1"] + df["MS2"]) / 2

avg_unlab_0   = (df[df["Target_Ox"]==0]["Unlabeled_C"] / df[df["Target_Ox"]==0]["Total_C"]).mean()
avg_unlab_100 = (df[df["Target_Ox"]==100]["Unlabeled_C"] / df[df["Target_Ox"]==100]["Total_C"]).mean()
df["Exp_Unlabeled"] = ((100 - df["Target_Ox"])*avg_unlab_0 + df["Target_Ox"]*avg_unlab_100) / 100
df["Expected"]      = (df["Target_Ox"]/100) * (1 - df["Exp_Unlabeled"])

for col in ESTIMATORS:
    df[f"Res_{col}"] = df[col] - df["Expected"]

# Summary stats per level
stats = df.groupby("Target_Ox").agg(
    MS1_mean=("MS1","mean"),      MS1_std=("MS1","std"),
    MS2_mean=("MS2","mean"),      MS2_std=("MS2","std"),
    Comb_mean=("Combined","mean"),Comb_std=("Combined","std"),
    Exp_mean=("Expected","mean"),
).reset_index()
for col, std, mean in [("MS1","MS1_std","MS1_mean"),
                        ("MS2","MS2_std","MS2_mean"),
                        ("Combined","Comb_std","Comb_mean")]:
    stats[f"{col}_CV"] = (stats[std] / stats[mean]) * 100

# Global performance metrics
def safe_mape(obs, exp):
    mask = exp != 0
    return np.mean(np.abs((obs[mask]-exp[mask])/exp[mask]))*100

metrics = {}
for col in ESTIMATORS:
    obs = df[col].values; exp = df["Expected"].values; res = obs - exp
    metrics[col] = dict(
        R2   = r2_score(exp, obs),
        RMSE = np.sqrt(mean_squared_error(exp, obs)),
        MAE  = mean_absolute_error(exp, obs),
        MAPE = safe_mape(obs, exp),
        Bias = np.mean(res),
        SD   = np.std(res),
    )

# Replicate wide table (Combined estimator)
df["Rep"] = df.groupby("Target_Ox").cumcount() + 1
wide = df.pivot_table(index="Target_Ox", columns="Rep", values="Combined")
wide.columns = ["Rep1","Rep2","Rep3"]
wide = wide.reset_index()

# ── Helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.grid(True, color=GRID_COL, linewidth=0.35, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

def panel_label(ax, label, x=-0.15, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")

bar_w = 0.52
x_est = np.arange(len(ESTIMATORS))

# ── Figure & GridSpec ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 9.6))

# Four row bands with explicit vertical positions
gs1 = gridspec.GridSpec(1, 2, figure=fig,
                         left=0.09, right=0.97, top=0.980, bottom=0.780,
                         wspace=0.36)
gs2 = gridspec.GridSpec(1, 2, figure=fig,
                         left=0.09, right=0.97, top=0.748, bottom=0.548,
                         wspace=0.36)
gs3 = gridspec.GridSpec(1, 4, figure=fig,
                         left=0.09, right=0.97, top=0.508, bottom=0.345,
                         wspace=0.62)
gs4 = gridspec.GridSpec(1, 3, figure=fig,
                         left=0.09, right=0.97, top=0.305, bottom=0.068,
                         wspace=0.40)

ax_A = fig.add_subplot(gs1[0,0])
ax_B = fig.add_subplot(gs1[0,1])
ax_C = fig.add_subplot(gs2[0,0])
ax_D = fig.add_subplot(gs2[0,1])
ax_E = fig.add_subplot(gs3[0,0])
ax_F = fig.add_subplot(gs3[0,1])
ax_G = fig.add_subplot(gs3[0,2])
ax_H = fig.add_subplot(gs3[0,3])
ax_I = fig.add_subplot(gs4[0,0])
ax_J = fig.add_subplot(gs4[0,1])
ax_K = fig.add_subplot(gs4[0,2])

ox_vals = np.array(LEVEL_ORDER)

# ════════════════════════════════════════════════════════════════════════════
# A — Calibration curve (mean ± SD)
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_A)
panel_label(ax_A, "A", x=-0.13)

mean_cols = {"MS1":"MS1_mean","MS2":"MS2_mean","Combined":"Comb_mean"}
std_cols  = {"MS1":"MS1_std", "MS2":"MS2_std", "Combined":"Comb_std"}

for col, color in zip(ESTIMATORS, EST_COLORS):
    means = stats[mean_cols[col]].values
    sds   = stats[std_cols[col]].values
    ax_A.fill_between(ox_vals, means-sds, means+sds,
                      color=color, alpha=0.13, zorder=1)
    ax_A.plot(ox_vals, means, color=color,
              marker=EST_MARKERS[ESTIMATORS.index(col)],
              markersize=4, linewidth=1.0,
              linestyle=DASHES[col], label=col, zorder=3)

ax_A.plot(ox_vals, stats["Exp_mean"].values,
          color=EXPECT_COL, linewidth=0.8, linestyle=(0,(6,3)),
          label="Expected", zorder=2)

ax_A.set_xlabel("Cysteine oxidation, nominal (%)")
ax_A.set_ylabel("Reduced fraction (computed)")
ax_A.set_xlim(-3, 103); ax_A.set_ylim(-0.02, 1.05)
ax_A.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax_A.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

leg_A = [
    Line2D([0],[0], color=c, marker=m, markersize=4,
           linewidth=1.0, linestyle=DASHES[e], label=e)
    for e,c,m in zip(ESTIMATORS, EST_COLORS, EST_MARKERS)
] + [
    Line2D([0],[0], color=EXPECT_COL, linewidth=0.8,
           linestyle=(0,(6,3)), label="Expected"),
    Patch(facecolor=MS1_COL, alpha=0.15, edgecolor="none", label="±1 SD"),
]
ax_A.legend(handles=leg_A, frameon=False, loc="upper left",
            handlelength=2.0, handletextpad=0.5, labelspacing=0.3)

# ════════════════════════════════════════════════════════════════════════════
# B — Per-site scatter onto identity
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_B)
panel_label(ax_B, "B", x=-0.13)

for col, marker in zip(ESTIMATORS, EST_MARKERS):
    for _, row in df.iterrows():
        ax_B.scatter(row["Expected"], row[col],
                     color=PALETTE[row["Target_Ox"]], marker=marker,
                     s=16, alpha=0.88, linewidths=0.3,
                     edgecolors="white", zorder=3)

lim = (-0.02, 1.05)
ax_B.plot(lim, lim, color=EXPECT_COL, linewidth=0.8,
          linestyle=(0,(6,3)), zorder=2)
ax_B.set_xlabel("Expected reduced fraction")
ax_B.set_ylabel("Observed reduced fraction")
ax_B.set_xlim(*lim); ax_B.set_ylim(*lim)
ax_B.set_aspect("equal")

lev_handles = [
    Line2D([0],[0], marker="o", color="none",
           markerfacecolor=PALETTE[ox], markeredgecolor="none",
           markersize=5, label=f"{ox}%")
    for ox in LEVEL_ORDER
]
shp_handles = [
    Line2D([0],[0], marker=m, color="none",
           markerfacecolor="#888", markeredgecolor="none",
           markersize=5, label=e)
    for m,e in zip(EST_MARKERS, ESTIMATORS)
]
leg_lev = ax_B.legend(handles=lev_handles, title="Nominal level",
                      title_fontsize=5.5, frameon=False,
                      loc="upper left", fontsize=5.5,
                      handlelength=0.8, labelspacing=0.2, ncol=2)
ax_B.add_artist(leg_lev)
ax_B.legend(handles=shp_handles, title="Estimator",
            title_fontsize=5.5, frameon=False,
            loc="lower right", fontsize=5.5,
            handlelength=0.8, labelspacing=0.2)

# ════════════════════════════════════════════════════════════════════════════
# C — Residual scatter
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_C)
panel_label(ax_C, "C", x=-0.13)

ax_C.axhspan(-0.05, 0.05, color=BAND_COL, zorder=0)
ax_C.axhline(0, color=EXPECT_COL, linewidth=0.7,
             linestyle=(0,(5,3)), zorder=2)

for col, color, marker in zip(ESTIMATORS, EST_COLORS, EST_MARKERS):
    ax_C.scatter(df["Expected"], df[f"Res_{col}"],
                 color=color, marker=marker, s=16, alpha=0.85,
                 linewidths=0.3, edgecolors="white", zorder=3, label=col)

ax_C.set_xlabel("Expected reduced fraction")
ax_C.set_ylabel("Residual (observed − expected)")
ax_C.set_xlim(-0.02, 1.05); ax_C.set_ylim(-0.10, 0.10)
ax_C.yaxis.set_major_locator(mticker.MultipleLocator(0.025))

leg_C = [
    Line2D([0],[0], marker=m, color="none", markerfacecolor=c,
           markersize=5, label=e)
    for m,c,e in zip(EST_MARKERS, EST_COLORS, ESTIMATORS)
] + [Patch(facecolor=BAND_COL, edgecolor=GRID_COL, lw=0.4, label="±5% tolerance")]
ax_C.legend(handles=leg_C, frameon=False, loc="upper right",
            handlelength=1.2, handletextpad=0.5, labelspacing=0.3)

# ════════════════════════════════════════════════════════════════════════════
# D — CV% by level (0% excluded)
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_D)
panel_label(ax_D, "D", x=-0.13)

stats_cv   = stats[stats["Target_Ox"] != 0].reset_index(drop=True)
levels_cv  = stats_cv["Target_Ox"].values
x_cv = np.arange(len(levels_cv))
w_cv = 0.26

for i, (col_cv, color, label) in enumerate(zip(
        ["MS1_CV","MS2_CV","Combined_CV"], EST_COLORS, ESTIMATORS)):
    ax_D.bar(x_cv + (i-1)*w_cv, stats_cv[col_cv].values,
             width=w_cv*0.88, color=color, alpha=0.85,
             label=label, zorder=3, linewidth=0)

ax_D.axhline(5, color="#888888", linewidth=0.7,
             linestyle=(0,(4,2)), zorder=4, label="5% threshold")
ax_D.set_xticks(x_cv)
ax_D.set_xticklabels([f"{ox}%" for ox in levels_cv])
ax_D.set_xlabel("Nominal oxidation level")
ax_D.set_ylabel("CV (%)")
max_cv = stats_cv[["MS1_CV","MS2_CV","Combined_CV"]].values.max()
ax_D.set_ylim(0, max_cv * 1.30)
ax_D.legend(frameon=False, loc="upper right",
            handlelength=1.0, handletextpad=0.5, labelspacing=0.3)

# ════════════════════════════════════════════════════════════════════════════
# E — R² (zoomed)
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_E)
panel_label(ax_E, "E", x=-0.24, y=1.18)

r2_vals = [metrics[e]["R2"] for e in ESTIMATORS]
bars_E  = ax_E.bar(x_est, r2_vals, width=bar_w,
                   color=EST_COLORS, alpha=0.88, zorder=3, linewidth=0)
for bar, val in zip(bars_E, r2_vals):
    ax_E.text(bar.get_x()+bar.get_width()/2, val+0.00004,
              f"{val:.4f}", ha="center", va="bottom",
              fontsize=5.0, color=EXPECT_COL)

ax_E.set_xticks(x_est)
ax_E.set_xticklabels(ESTIMATORS, fontsize=5.5)
ax_E.set_ylabel("R²"); ax_E.set_ylim(0.990, 1.0005)
ax_E.yaxis.set_major_locator(mticker.MultipleLocator(0.002))
ax_E.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax_E.set_title("R²", fontsize=7, pad=6)
ax_E.spines["bottom"].set_visible(False)
ax_E.tick_params(bottom=False)
ax_E.set_xlim(-0.6, len(ESTIMATORS)-0.4)

# ════════════════════════════════════════════════════════════════════════════
# F — RMSE / MAE / SD
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_F)
panel_label(ax_F, "F", x=-0.24, y=1.18)

metric_keys = ["RMSE","MAE","SD"]
metric_labs = ["RMSE","MAE","SD(resid.)"]
x_mf  = np.arange(len(metric_keys))
w_mf  = 0.22

for i,(est,color) in enumerate(zip(ESTIMATORS, EST_COLORS)):
    vals = [metrics[est][k] for k in metric_keys]
    ax_F.bar(x_mf+(i-1)*w_mf, vals,
             width=w_mf*0.88, color=color, alpha=0.88,
             label=est, zorder=3, linewidth=0)

ax_F.set_xticks(x_mf)
ax_F.set_xticklabels(metric_labs, fontsize=5.5)
ax_F.set_ylabel("Reduced fraction")
ax_F.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax_F.set_ylim(0, max(metrics[e][k] for e in ESTIMATORS
                     for k in metric_keys)*1.38)
ax_F.set_title("Error metrics", fontsize=7, pad=6)
ax_F.legend(frameon=False, loc="upper right",
            handlelength=0.8, handletextpad=0.4, labelspacing=0.25)

# ════════════════════════════════════════════════════════════════════════════
# G — Bias (diverging)
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_G)
panel_label(ax_G, "G", x=-0.28, y=1.18)

bias_vals = [metrics[e]["Bias"] for e in ESTIMATORS]
for xi,(val,color) in enumerate(zip(bias_vals, EST_COLORS)):
    ax_G.bar(xi, val, width=bar_w, color=color, alpha=0.88,
             zorder=3, linewidth=0)
    ax_G.text(xi, val+(0.0004 if val>=0 else -0.0004),
              f"{val:+.4f}", ha="center",
              va="bottom" if val>=0 else "top",
              fontsize=5.0, color=EXPECT_COL)

ax_G.axhline(0, color=EXPECT_COL, linewidth=0.6, zorder=4)
ax_G.set_xticks(x_est)
ax_G.set_xticklabels(ESTIMATORS, fontsize=5.5)
ax_G.set_ylabel("Bias (obs. − exp.)")
ax_G.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax_G.set_title("Systematic bias", fontsize=7, pad=6)
bmax = max(abs(v) for v in bias_vals)
ax_G.set_ylim(-bmax*1.8, bmax*1.8)
ax_G.set_xlim(-0.6, len(ESTIMATORS)-0.4)

# ════════════════════════════════════════════════════════════════════════════
# H — MAPE%
# ════════════════════════════════════════════════════════════════════════════
style_ax(ax_H)
panel_label(ax_H, "H", x=-0.28, y=1.18)

mape_vals = [metrics[e]["MAPE"] for e in ESTIMATORS]
bars_H = ax_H.bar(x_est, mape_vals, width=bar_w,
                  color=EST_COLORS, alpha=0.88, zorder=3, linewidth=0)
for bar,val in zip(bars_H, mape_vals):
    ax_H.text(bar.get_x()+bar.get_width()/2, val+0.09,
              f"{val:.2f}%", ha="center", va="bottom",
              fontsize=5.0, color=EXPECT_COL)

ax_H.axhline(5, color="#888888", linewidth=0.7,
             linestyle=(0,(4,2)), zorder=4, label="5% threshold")
ax_H.set_xticks(x_est)
ax_H.set_xticklabels(ESTIMATORS, fontsize=5.5)
ax_H.set_ylabel("MAPE (%)")
ax_H.set_ylim(0, max(mape_vals)*1.38)
ax_H.set_title("MAPE", fontsize=7, pad=6)
ax_H.legend(frameon=False, loc="upper right",
            handlelength=1.0, handletextpad=0.4)

# ════════════════════════════════════════════════════════════════════════════
# I, J, K — Technical replicate correlations (Combined estimator)
# ════════════════════════════════════════════════════════════════════════════
pairs      = [("Rep1","Rep2"),("Rep1","Rep3"),("Rep2","Rep3")]
pair_labs  = ["Rep 1 vs. Rep 2","Rep 1 vs. Rep 3","Rep 2 vs. Rep 3"]
panel_labs = ["I","J","K"]

for ax, (r1,r2), plabel, plab in zip(
        [ax_I, ax_J, ax_K], pairs, pair_labs, panel_labs):
    style_ax(ax)
    panel_label(ax, plab, x=-0.18)

    for ox in LEVEL_ORDER:
        row = wide[wide["Target_Ox"]==ox]
        ax.scatter(row[r1], row[r2],
                   color=PALETTE[ox], s=40, alpha=0.95,
                   edgecolors="white", linewidths=0.5, zorder=3,
                   label=f"{ox}%")

    qlim = (-0.02, 1.05)
    ax.plot(qlim, qlim, color=EXPECT_COL,
            linewidth=0.7, linestyle=(0,(6,3)), zorder=2)

    x_all = wide[r1].values
    y_all = wide[r2].values
    r_val, _ = pearsonr(x_all, y_all)
    r2_val   = r2_score(y_all, x_all)
    ax.text(0.05, 0.94,
            f"r = {r_val:.4f}\nR² = {r2_val:.4f}",
            transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRID_COL, linewidth=0.5))

    ax.set_xlim(*qlim); ax.set_ylim(*qlim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{r1} reduced fraction")
    ax.set_ylabel(f"{r2} reduced fraction")
    ax.set_title(plabel, fontsize=7, pad=4)

# Shared level colour legend below row 4
lev_handles_bot = [
    Line2D([0],[0], marker="o", color="none",
           markerfacecolor=PALETTE[ox], markeredgecolor="none",
           markersize=5, label=f"{ox}%")
    for ox in LEVEL_ORDER
]
fig.legend(handles=lev_handles_bot,
           title="Nominal oxidation level",
           title_fontsize=6,
           loc="lower center",
           ncol=len(LEVEL_ORDER),
           frameon=False,
           fontsize=6,
           bbox_to_anchor=(0.53, 0.0),
           handlelength=0.8,
           columnspacing=0.8,
           handletextpad=0.4)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig("fig_grand.png",  dpi=600)
fig.savefig("fig_grand.pdf")
print("✓  fig_grand.png  (600 DPI)")
print("✓  fig_grand.pdf  (vector)")
