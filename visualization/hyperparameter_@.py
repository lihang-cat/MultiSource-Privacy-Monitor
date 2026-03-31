import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick

# =========================
# 0. Publication-grade style (Extra Large Fonts)
# =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # --- 修改处：全线大幅调大字号 ---
    "font.size": 18,            # 原 15 -> 18
    "axes.labelsize": 24,       # 原 18 -> 24 (非常醒目)
    "axes.titlesize": 24,       # 原 18 -> 24
    "xtick.labelsize": 20,      # 原 15 -> 20
    "ytick.labelsize": 20,      # 原 15 -> 20
    "legend.fontsize": 20,      # 原 14 -> 20
    "axes.linewidth": 1.5,      # 保持适中
    "grid.linewidth": 0.8,      # 保持适中
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DPI = 600

# =========================
# 1. Data (保持不变)
# =========================
ratios = [0.3, 0.4, 0.5, 0.8, 0.9, 1.0]
scores = {
    "α = 0.0 (None)":    [0.4896, 0.7518, 0.7633, 0.6164, 0.8368, 0.8390],
    "α = 0.5 (Weak)":    [0.5325, 0.7938, 0.8152, 0.7194, 0.8904, 0.9097],
    "α = 0.8 (Optimal)": [0.5568, 0.8158, 0.8423, 0.7613, 0.9045, 0.9333],
    "α = 1.5 (Strong)":  [0.6093, 0.8584, 0.8948, 0.8011, 0.8995, 0.9446],
    "α = 3.0 (Extreme)": [0.7018, 0.9134, 0.9615, 0.6131, 0.8285, 0.8545],
}

rows = []
for alpha, vals in scores.items():
    for r, s in zip(ratios, vals):
        rows.append({"Ratio": r, "Score": s, "Alpha": alpha})

df = pd.DataFrame(rows)

# =========================
# 2. Figure
# =========================
# 稍微加大画布以容纳大字体
fig, ax = plt.subplots(figsize=(11, 8), dpi=DPI)
sns.set_theme(style="whitegrid", rc={"grid.color": ".92", "grid.linestyle": "--"})

palette = {
    "α = 0.0 (None)":    "#9aa0a6",
    "α = 0.5 (Weak)":    "#4c72b0",
    "α = 0.8 (Optimal)": "#c0392b",
    "α = 1.5 (Strong)":  "#dd8452",
    "α = 3.0 (Extreme)": "#8172b3",
}

linestyles = {
    "α = 0.0 (None)":    (0, (3, 3)),
    "α = 0.5 (Weak)":    (0, (3, 3)),
    "α = 0.8 (Optimal)": "solid",
    "α = 1.5 (Strong)":  (0, (5, 3)),
    "α = 3.0 (Extreme)": (0, (1, 2)),
}

# 保持您原设定的线宽 (Optimal=3.5 很粗了，足够匹配大字体)
linewidths = {
    "α = 0.8 (Optimal)": 3.5,
}
default_linewidth = 2.0

for alpha, g in df.groupby("Alpha"):
    ax.plot(
        g["Ratio"], g["Score"],
        label=alpha,
        color=palette[alpha],
        linestyle=linestyles[alpha],
        linewidth=linewidths.get(alpha, default_linewidth),
        marker="o",
        markersize=12 if alpha == "α = 0.8 (Optimal)" else 9,
        markerfacecolor="white",
        markeredgewidth=1.5
    )

# =========================
# 3. Critical threshold
# =========================
ax.axhline(
    y=0.90,
    color="#c0392b",
    linestyle="--",
    linewidth=1.8,
    alpha=0.7,
    zorder=0
)

# =========================
# 4. Key narrative annotation (大字体)
# =========================
ax.annotate(
    "Optimal Balance\n(Stable & Robust)",
    xy=(1.0, 0.9333),
    xytext=(0.68, 1.08), # 位置微调
    arrowprops=dict(arrowstyle="->", lw=2.5, color="#c0392b"), # 箭头稍微加粗
    fontsize=20,          # 原 15 -> 20
    fontweight="bold",
    color="#c0392b",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c0392b", linewidth=1.5)
)

# =========================
# 5. Axis & output
# =========================
ax.set_xlabel("Sensitivity Ratio ($R_i$)", fontweight="bold", labelpad=10)
ax.set_ylabel("Total Score ($TS$)", fontweight="bold", labelpad=10)

ax.set_xlim(0.28, 1.05)
ax.set_ylim(0.45, 1.18) # 稍微增加顶部空间给大字体的注释

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

# 增强刻度标签的粗细
plt.setp(ax.get_xticklabels(), fontweight="bold")
plt.setp(ax.get_yticklabels(), fontweight="bold")

leg = ax.legend(
    title="$\\alpha$ Configuration",
    loc="lower left",
    frameon=True,
    fancybox=True,
    fontsize=20,        # 内容字体 -> 20
    title_fontsize=22,  # 标题字体 -> 22
    handlelength=2.5    # 稍微拉长图例线
)
leg.get_title().set_fontweight("bold")

sns.despine(trim=True)
plt.tight_layout()

plt.savefig("Fig5_Sensitivity_ExtraLargeFonts.pdf", bbox_inches="tight")
plt.savefig("Fig5_Sensitivity_ExtraLargeFonts.png", dpi=DPI, bbox_inches="tight")
plt.show()