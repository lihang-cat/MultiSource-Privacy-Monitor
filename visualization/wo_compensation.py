import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

# ================= 0. 全局设置 (特大字体 - Extra Large Fonts) =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 保持字体特大 ---
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 24      # 轴标题：极大
plt.rcParams['xtick.labelsize'] = 20     # 刻度：极大
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20     # 图例：极大
plt.rcParams['axes.linewidth'] = 1.5     # 边框恢复适中
plt.rcParams['grid.linewidth'] = 0.8     # 网格恢复适中
Target_DPI = 600

# ================= 1. 数据录入 (保持不变) =================
data = [
    # --- 1. Full Model ---
    {"Group": "g3",  "Ratio": 0.30, "Score": 0.5568, "Model": "Full Model (Optimized)"},
    {"Group": "g7",  "Ratio": 0.40, "Score": 0.8158, "Model": "Full Model (Optimized)"},
    {"Group": "g11", "Ratio": 0.50, "Score": 0.8423, "Model": "Full Model (Optimized)"},
    {"Group": "g4",  "Ratio": 0.80, "Score": 0.7613, "Model": "Full Model (Optimized)"},
    {"Group": "g8",  "Ratio": 0.90, "Score": 0.9045, "Model": "Full Model (Optimized)"},
    {"Group": "g12", "Ratio": 1.00, "Score": 0.9333, "Model": "Full Model (Optimized)"},

    # --- 2. Ablated Model ---
    {"Group": "g3",  "Ratio": 0.30, "Score": 0.4896, "Model": "Ablated (w/o Compensation)"},
    {"Group": "g7",  "Ratio": 0.40, "Score": 0.7518, "Model": "Ablated (w/o Compensation)"},
    {"Group": "g11", "Ratio": 0.50, "Score": 0.7633, "Model": "Ablated (w/o Compensation)"},
    {"Group": "g4",  "Ratio": 0.80, "Score": 0.6164, "Model": "Ablated (w/o Compensation)"},
    {"Group": "g8",  "Ratio": 0.90, "Score": 0.8368, "Model": "Ablated (w/o Compensation)"},
    {"Group": "g12", "Ratio": 1.00, "Score": 0.8390, "Model": "Ablated (w/o Compensation)"},
]

df = pd.DataFrame(data)

# ================= 2. 绘图设置 =================
fig, ax = plt.subplots(figsize=(11, 8), dpi=Target_DPI)
sns.set_theme(style="whitegrid", rc={"grid.color": ".9"})

palette = {"Full Model (Optimized)": "#c0392b", "Ablated (w/o Compensation)": "#8e44ad"}
styles = {"Full Model (Optimized)": "-", "Ablated (w/o Compensation)": "--"}

# --- 修改处：线条与点恢复原状 (不要太粗) ---
sns.lineplot(
    x="Ratio", y="Score", hue="Model", style="Model",
    data=df, palette=palette, markers=True,
    markersize=11,  # 恢复为 11 (之前是 14)
    linewidth=3.0,  # 恢复为 3.0 (之前是 4.0)
    ax=ax
)

# ================= 3. 关键标注 (字体保持特大) =================
# 阈值线
ax.axhline(y=0.9, color='#c0392b', linestyle=':', linewidth=1.5, alpha=0.8)
# fontsize 保持 18
ax.text(1.02, 0.9, ' Critical (0.9)', color='#c0392b', va='center', fontweight='bold', fontsize=18)

ax.axhline(y=0.7, color='#d35400', linestyle=':', linewidth=1.5, alpha=0.8)
# fontsize 保持 18
ax.text(1.02, 0.7, ' High (0.7)', color='#d35400', va='center', fontweight='bold', fontsize=18)

# 1. 标注 g12 的降级
# fontsize 保持 19
ax.annotate('Failure to Trigger Critical\n(0.93 $\\to$ 0.84)',
            xy=(1.0, 0.8390), xytext=(0.55, 0.52),
            arrowprops=dict(facecolor='#8e44ad', shrink=0.05, width=1.5, headwidth=8, connectionstyle="arc3,rad=0.2"),
            fontsize=19, color='#8e44ad', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#8e44ad", alpha=0.9, linewidth=1.5))

# 2. 标注 g4 的降级
# fontsize 保持 19
ax.annotate('Risk Underestimation\n(High $\\to$ Medium)',
            xy=(0.8, 0.6164), xytext=(0.82, 0.42),
            arrowprops=dict(facecolor='#8e44ad', shrink=0.05, width=1.5, headwidth=8),
            fontsize=19, color='#8e44ad', fontweight='bold')

# 3. 填充差距区域
full_scores = df[df["Model"]=="Full Model (Optimized)"]["Score"].values
abl_scores = df[df["Model"]=="Ablated (w/o Compensation)"]["Score"].values
ratios = df[df["Model"]=="Full Model (Optimized)"]["Ratio"].values
ax.fill_between(ratios[3:], abl_scores[3:], full_scores[3:], color='gray', alpha=0.15, label='Performance Gap')


# ================= 4. 美化与图例调整 =================
ax.set_xlabel("Sensitivity Ratio ($R_i$)", fontweight='bold', labelpad=10)
ax.set_ylabel("Total Score ($TS$)", fontweight='bold', labelpad=10)
ax.set_ylim(0.4, 1.05)
ax.set_xlim(0.25, 1.05)

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# 图例设置：字体极大(20)，但线条图示保持优雅
plt.legend(
    loc='upper left',
    frameon=True,
    fontsize=20,
    title="",
    handlelength=2.0, # 恢复正常长度
    handleheight=1.0
)

plt.tight_layout()

plt.savefig('Experiment2b_Ablation_ExtraLargeFonts_NormalLines.png', dpi=Target_DPI)
plt.savefig('Experiment2b_Ablation_ExtraLargeFonts_NormalLines.pdf', dpi=Target_DPI)
plt.show()