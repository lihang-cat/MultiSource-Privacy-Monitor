
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

# =========================
# 0. 环境设置
# =========================
sns.set_theme(style="whitegrid", rc={"grid.color": ".90", "grid.linestyle": "--"})
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
beta_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
tau_x = [1, 2, 3, 4, 5, 6]
lambda_x = [0.1, 0.3, 0.5, 0.7, 0.9]

data_beta = {
    "G12 ": [np.float64(0.8057), np.float64(0.8774), np.float64(0.9166), np.float64(0.931), np.float64(0.9285), np.float64(0.9169)],
    "G11 ": [np.float64(0.5669), np.float64(0.6901), np.float64(0.7798), np.float64(0.8411), np.float64(0.8791), np.float64(0.8989)],
    "G10 ": [np.float64(0.13), np.float64(0.2349), np.float64(0.3305), np.float64(0.417), np.float64(0.4942), np.float64(0.5622)],
    "G8  ": [np.float64(0.7076), np.float64(0.7966), np.float64(0.8535), np.float64(0.8852), np.float64(0.8987), np.float64(0.9009)],
    "G7  ": [np.float64(0.4537), np.float64(0.5947), np.float64(0.7025), np.float64(0.7815), np.float64(0.8362), np.float64(0.871)],
    "G6  ": [np.float64(0.2231), np.float64(0.3971), np.float64(0.5385), np.float64(0.6507), np.float64(0.7371), np.float64(0.801)],
}

data_tau = {
    "G12 ": [np.float64(0.9166), np.float64(0.9166), np.float64(0.9166), np.float64(0.9166), np.float64(0.9166), np.float64(0.9166)],
    "G11 ": [np.float64(0.7798), np.float64(0.7798), np.float64(0.7798), np.float64(0.7798), np.float64(0.7798), np.float64(0.7798)],
    "G10 ": [np.float64(0.3838), np.float64(0.3838), np.float64(0.3838), np.float64(0.3305), np.float64(0.3127), np.float64(0.3127)],
    "G8  ": [np.float64(0.8535), np.float64(0.8535), np.float64(0.8535), np.float64(0.8535), np.float64(0.8535), np.float64(0.8535)],
    "G7  ": [np.float64(0.7025), np.float64(0.7025), np.float64(0.7025), np.float64(0.7025), np.float64(0.7025), np.float64(0.7025)],
    "G6  ": [np.float64(0.5385), np.float64(0.5385), np.float64(0.5385), np.float64(0.5385), np.float64(0.5385), np.float64(0.5385)],
}

data_lambda = {
    "G12 ": [np.float64(0.8351), np.float64(0.8758), np.float64(0.9166), np.float64(0.9573), np.float64(0.998)],
    "G11 ": [np.float64(0.7105), np.float64(0.7451), np.float64(0.7798), np.float64(0.8145), np.float64(0.8491)],
    "G10 ": [np.float64(0.339), np.float64(0.3348), np.float64(0.3305), np.float64(0.3262), np.float64(0.322)],
    "G8  ": [np.float64(0.7777), np.float64(0.8156), np.float64(0.8535), np.float64(0.8915), np.float64(0.9294)],
    "G7  ": [np.float64(0.64), np.float64(0.6712), np.float64(0.7025), np.float64(0.7337), np.float64(0.7649)],
    "G6  ": [np.float64(0.4907), np.float64(0.5146), np.float64(0.5385), np.float64(0.5625), np.float64(0.5864)],
}

# 【核心修改 1】赋予完全独立的形状 (Marker)
STYLE_MAP = {
    "G12 ": {"color": "#c0392b", "marker": "s", "ls": "solid"},  # 4源高危：实线 红 正方
    "G11 ": {"color": "#e67e22", "marker": "o", "ls": "solid"},  # 4源中危：实线 橙 圆形
    "G10 ": {"color": "#2980b9", "marker": "^", "ls": "solid"},  # 4源低危：实线 蓝 上三角

    "G8  ": {"color": "#c0392b", "marker": "D", "ls": "--"},  # 3源高危：虚线 红 菱形
    "G7  ": {"color": "#e67e22", "marker": "p", "ls": "--"},  # 3源中危：虚线 橙 五边
    "G6  ": {"color": "#2980b9", "marker": "v", "ls": "--"},  # 3源低危：虚线 蓝 下三角
}

OPT_IDX = {"beta": 2, "tau": 3, "lambda": 2}  # 默认最优点

# =========================
# 2. 绘图执行
# =========================
fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), dpi=600)


def plot_multiline(ax, x_vals, data_dict, xlabel, title, opt_idx, is_int=False):
    for group_name, y_vals in data_dict.items():
        style = STYLE_MAP[group_name]
        ax.plot(x_vals, y_vals, label=group_name, color=style["color"],
                linestyle=style["ls"], lw=3.5, marker=style["marker"],
                ms=10, mfc="white", mew=2.5, zorder=2)

    opt_x = x_vals[opt_idx]
    ax.axvline(x=opt_x, color="gray", linestyle="-.", linewidth=2.5, alpha=0.6, zorder=1)
    ax.text(opt_x, 1.05, "Optimal\nSetting", ha='center', va='bottom',
            fontsize=14, fontweight='bold', color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f1c40f", ec="none", alpha=0.9))

    ax.set_title(title, fontweight="bold", pad=35)
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=12)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    if is_int: ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")


# 【核心修改 2】使用 \mathrm{} 修复所有正体角标排版
plot_multiline(axes[0], beta_x, data_beta, r"Base Sensitivity Coeff. ($\beta$)", r"(a) Effect of $\beta$",
               OPT_IDX["beta"])
axes[0].set_ylabel("Total Score ($TS$)", fontweight="bold", labelpad=10)

plot_multiline(axes[1], tau_x, data_tau, r"Saturation Threshold ($\tau_{\mathrm{sat}}$)",
               r"(b) Effect of $\tau_{\mathrm{sat}}$",
               OPT_IDX["tau"], is_int=True)

plot_multiline(axes[2], lambda_x, data_lambda, r"Dynamic Weight ($\lambda_{\mathrm{dyn}}$)",
               r"(c) Effect of $\lambda_{\mathrm{dyn}}$",
               OPT_IDX["lambda"])


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.18),
           ncol=3, fontsize=16, frameon=True, fancybox=True, shadow=True,
           labelspacing=0.8, columnspacing=2.0, handlelength=3.5)

plt.subplots_adjust(wspace=0.2)
plt.tight_layout()
plt.savefig("Fig6_Hyperparams_Final.pdf", bbox_inches="tight")
print("✅ SCI 顶刊终极完美版图表绘制完成！")
plt.show()