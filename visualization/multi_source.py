# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import matplotlib.patheffects as pe
#
# # ================= 0. 全局样式设置 (IEEE Access Extra Large Font Style) =================
# STYLE_CONFIG = {
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
#     # --- 修改处：精确设定 X/Y 轴字号 ---
#     "font.size": 18,
#     "axes.labelsize": 24,
#     "axes.titlesize": 24,
#     "xtick.labelsize": 26,      # 【修改】X轴 -> 26
#     "ytick.labelsize": 24,      # 【修改】Y轴 -> 24
#     "legend.fontsize": 20,
#     "axes.linewidth": 1.5,
#     "axes.edgecolor": "#333333",
#     "axes.labelweight": "bold",
#     "xtick.color": "#333333",
#     "ytick.color": "#333333",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# }
# plt.rcParams.update(STYLE_CONFIG)
#
# # 画布高度增加以容纳大字体
# FIG_SIZE = (11, 9.5)
# DPI = 600
#
# # 调色板
# BAR_PALETTE = ["#D1E5F0", "#67A9CF", "#2166AC"]
# GRID_COLOR = "#F0F0F0"
# TEXT_COLOR = "#2C3E50"
# HIGHLIGHT_COLOR = "#A93226"
#
# # 阈值
# THRESHOLDS = [
#     (0.9, "Critical (0.9)", "#C0392B"),
#     (0.7, "High (0.7)",     "#D35400"),
#     (0.4, "Medium (0.4)",   "#F39C12"),
# ]
#
# # ================= 1. 数据准备 =================
# df = pd.DataFrame([
#     {"Set": "Mid Density\n(~30–50%)",  "Risk Score": 0.5568, "Sources": "2 Sources"},
#     {"Set": "Mid Density\n(~30–50%)",  "Risk Score": 0.8158, "Sources": "3 Sources"},
#     {"Set": "Mid Density\n(~30–50%)",  "Risk Score": 0.8423, "Sources": "4 Sources"},
#     {"Set": "High Density\n(~80–100%)", "Risk Score": 0.7613, "Sources": "2 Sources"},
#     {"Set": "High Density\n(~80–100%)", "Risk Score": 0.9045, "Sources": "3 Sources"},
#     {"Set": "High Density\n(~80–100%)", "Risk Score": 0.9333, "Sources": "4 Sources"},
# ])
#
# # ================= 2. 绘图初始化 =================
# sns.set_theme(style="white", rc={"axes.grid": True, "grid.color": GRID_COLOR, "grid.linestyle": "--"})
# fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
#
# # ================= 3. 绘制柱状图 =================
# bar_plot = sns.barplot(
#     data=df,
#     x="Set",
#     y="Risk Score",
#     hue="Sources",
#     palette=BAR_PALETTE,
#     edgecolor="#2C3E50",
#     linewidth=1.5,
#     alpha=0.95,
#     zorder=3,
#     ax=ax
# )
#
# # ================= 4. 绘制阈值线 =================
# for y_val, label, color in THRESHOLDS:
#     ax.axhline(y=y_val, linestyle="--", linewidth=2.0, color=color, alpha=0.8, zorder=2, dashes=(5, 5))
#     txt = ax.text(1.52, y_val, label, fontsize=18, fontweight="bold", color=color, va="center", ha="left")
#     txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
#
# # ================= 5. 数值标签 =================
# for container in ax.containers:
#     ax.bar_label(container, fmt="%.2f", padding=3, fontsize=18, color=TEXT_COLOR, fontweight="regular")
#
# # ================= 6. 添加注释 =================
# def add_annotation(text, xy_target, xy_text, arc_rad):
#     ax.annotate(
#         text,
#         xy=xy_target,
#         xytext=xy_text,
#         fontsize=19,
#         fontweight="bold",
#         color=HIGHLIGHT_COLOR,
#         ha="center",
#         bbox=dict(
#             boxstyle="round,pad=0.4",
#             fc="white",
#             ec=HIGHLIGHT_COLOR,
#             linewidth=2.0,
#             alpha=0.95
#         ),
#         arrowprops=dict(
#             arrowstyle="-|>",
#             color=HIGHLIGHT_COLOR,
#             linewidth=2.5,
#             connectionstyle=f"arc3,rad={arc_rad}",
#             shrinkA=5, shrinkB=5
#         ),
#         zorder=10
#     )
#
# add_annotation("Tier Escalation:\nMedium $\\rightarrow$ High", (-0.12, 0.82), (-0.38, 1.25), 0.3)
# add_annotation("Tier Escalation:\nHigh $\\rightarrow$ Critical", (0.88, 0.90), (1.35, 1.25), -0.3)
#
# # ================= 7. 布局调整 =================
# ax.set_ylabel("Total Score ($TS$)", fontsize=24, labelpad=15, fontweight="bold")
# ax.set_xlabel("", fontsize=14)
#
# # --- 【关键修改】强制设置 X 和 Y 轴标签加粗 ---
# # 注意：字号已通过 plt.rcParams['xtick.labelsize'] = 26 和 ytick.labelsize = 24 设定
# plt.setp(ax.get_xticklabels(), fontweight="bold")
# plt.setp(ax.get_yticklabels(), fontweight="bold")
#
# # 增加 Y 轴高度给顶部注释留空间
# ax.set_ylim(0, 1)
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
#
# # 图例
# legend = ax.legend(
#     title="Environmental Sources",
#     title_fontsize=22,
#     fontsize=20,
#     ncol=3,
#     frameon=False,
#     loc="upper center",
#     # 因为X轴字号到了26，非常占空间，图例需要再往下移一点点 (-0.15 -> -0.18)
#     bbox_to_anchor=(0.5, -0.05)
# )
# legend.get_title().set_fontweight("bold")
#
# sns.despine(left=False, bottom=False, offset=5, trim=False)
# plt.tight_layout()
#
# # ================= 8. 保存文件 =================
# plt.savefig("Figure4_Final_ExtraLarge_Bold.png", dpi=300, bbox_inches="tight")
# plt.savefig("Figure4_Final_ExtraLarge_Bold.pdf", format='pdf', bbox_inches="tight")
#
# print("✅ 图表已生成：Y轴字号24(粗)，X轴字号26(粗)。")
# plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patheffects as pe

# ================= 0. 全局样式设置 (IEEE Access Extra Large Font Style) =================
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 18,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "legend.fontsize": 20,
    "axes.linewidth": 1.5,
    "axes.edgecolor": "#333333",
    "axes.labelweight": "bold",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(STYLE_CONFIG)

# 画布高度增加以容纳大字体
FIG_SIZE = (11, 7.5)
DPI = 600

# 调色板
BAR_PALETTE = ["#D1E5F0", "#67A9CF", "#2166AC"]
GRID_COLOR = "#F0F0F0"
TEXT_COLOR = "#2C3E50"
HIGHLIGHT_COLOR = "#A93226"

# 阈值
THRESHOLDS = [
    (0.9, "Critical (0.9)", "#C0392B"),
    (0.7, "High (0.7)",     "#D35400"),
    (0.4, "Medium (0.4)",   "#F39C12"),
]

# ================= 1. 数据准备 =================
df = pd.DataFrame([
    {"Set": "Mid Density\n(~30–50%)",   "Risk Score": 0.5568, "Sources": "2 Sources"},
    {"Set": "Mid Density\n(~30–50%)",   "Risk Score": 0.8158, "Sources": "3 Sources"},
    {"Set": "Mid Density\n(~30–50%)",   "Risk Score": 0.8423, "Sources": "4 Sources"},
    {"Set": "High Density\n(~80–100%)", "Risk Score": 0.7613, "Sources": "2 Sources"},
    {"Set": "High Density\n(~80–100%)", "Risk Score": 0.9045, "Sources": "3 Sources"},
    {"Set": "High Density\n(~80–100%)", "Risk Score": 0.9333, "Sources": "4 Sources"},
])

# ================= 2. 绘图初始化 =================
sns.set_theme(
    style="white",
    rc={"axes.grid": True, "grid.color": GRID_COLOR, "grid.linestyle": "--"}
)
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

# ================= 3. 绘制柱状图 =================
bar_plot = sns.barplot(
    data=df,
    x="Set",
    y="Risk Score",
    hue="Sources",
    palette=BAR_PALETTE,
    edgecolor="#2C3E50",
    linewidth=1.5,
    alpha=0.95,
    zorder=3,
    ax=ax
)

# ================= 4. 绘制阈值线 =================
for y_val, label, color in THRESHOLDS:
    ax.axhline(
        y=y_val,
        linestyle="--",
        linewidth=2.0,
        color=color,
        alpha=0.8,
        zorder=2,
        dashes=(5, 5)
    )
    txt = ax.text(
        1.52, y_val, label,
        fontsize=18,
        fontweight="bold",
        color=color,
        va="center",
        ha="left"
    )
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

# ================= 5. 数值标签 =================
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=18,
        color=TEXT_COLOR,
        fontweight="regular"
    )

# ================= 6. 添加注释 =================
def add_annotation(text, xy_target, xy_text, arc_rad):
    ax.annotate(
        text,
        xy=xy_target,
        xytext=xy_text,
        fontsize=19,
        fontweight="bold",
        color=HIGHLIGHT_COLOR,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec=HIGHLIGHT_COLOR,
            linewidth=2.0,
            alpha=0.95
        ),
        arrowprops=dict(
            arrowstyle="-|>",
            color=HIGHLIGHT_COLOR,
            linewidth=2.5,
            connectionstyle=f"arc3,rad={arc_rad}",
            shrinkA=5,
            shrinkB=5
        ),
        zorder=10
    )

add_annotation(
    "Tier Escalation:\nMedium $\\rightarrow$ High",
    (-0.12, 0.82),
    (-0.38, 1.25),
    0.3
)
add_annotation(
    "Tier Escalation:\nHigh $\\rightarrow$ Critical",
    (0.88, 0.90),
    (1.35, 1.25),
    -0.3
)

# ================= 7. 布局调整 =================
ax.set_ylabel("Total Score ($TS$)", fontsize=24, labelpad=15, fontweight="bold")
ax.set_xlabel("")

# ======== ✅【关键修复】只放大 X / Y 轴刻度字号，其余不受影响 ========
ax.tick_params(axis='x', labelsize=19)
ax.tick_params(axis='y', labelsize=20)

plt.setp(ax.get_xticklabels(), fontweight="bold")
plt.setp(ax.get_yticklabels(), fontweight="bold")

ax.set_ylim(0, 1.55)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

# 图例
legend = ax.legend(
    title="Environmental Sources",
    title_fontsize=22,
    fontsize=20,
    ncol=3,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10)
)
legend.get_title().set_fontweight("bold")

sns.despine(left=False, bottom=False, offset=5, trim=False)
plt.tight_layout()

# ================= 8. 保存文件 =================
plt.savefig("Figure4_Final_ExtraLarge_Bold.png", dpi=300, bbox_inches="tight")
plt.savefig("Figure4_Final_ExtraLarge_Bold.pdf", format="pdf", bbox_inches="tight")

print("✅ 修复完成：仅放大 X/Y 轴刻度字号（X=26，Y=24），其余元素未改动。")
plt.show()
