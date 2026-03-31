import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_sci_plotting_style():
    """设置SCI论文标准的绘图样式 - IEEE Access 风格 (字体增大版)"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        # --- 修改处：整体调大字号 ---
        'font.size': 14,           # 原 11 -> 14
        'axes.labelsize': 16,      # 原 13 -> 16
        'axes.titlesize': 16,      # 新增
        'xtick.labelsize': 14,     # 新增：保证刻度清楚
        'ytick.labelsize': 14,     # 新增：保证刻度清楚
        'legend.fontsize': 14,     # 新增：保证图例清楚
        # ------------------------
        'figure.dpi': 600,
        'savefig.dpi': 600
    })

def create_risk_score_dataframe():
    """创建风险分数数据的DataFrame (数据保持不变)"""
    data = [
        {"Group": "G1", "Type": "Safe", "Model": "Traditional Linear", "Score": 0.0026},
        {"Group": "G5", "Type": "Safe", "Model": "Traditional Linear", "Score": 0.0006},
        {"Group": "G9", "Type": "Safe", "Model": "Traditional Linear", "Score": 0.0015},
        {"Group": "G1", "Type": "Safe", "Model": "Optimized MS-SIDM (Ours)", "Score": 0.1021},
        {"Group": "G5", "Type": "Safe", "Model": "Optimized MS-SIDM (Ours)", "Score": 0.0891},
        {"Group": "G9", "Type": "Safe", "Model": "Optimized MS-SIDM (Ours)", "Score": 0.1089},
        {"Group": "G2", "Type": "Low-Leak", "Model": "Traditional Linear", "Score": 0.0435},
        {"Group": "G6", "Type": "Low-Leak", "Model": "Traditional Linear", "Score": 0.1203},
        {"Group": "G10", "Type": "Low-Leak", "Model": "Traditional Linear", "Score": 0.0602},
        {"Group": "G2", "Type": "Low-Leak", "Model": "Optimized MS-SIDM (Ours)", "Score": 0.3419},
        {"Group": "G6", "Type": "Low-Leak", "Model": "Optimized MS-SIDM (Ours)", "Score": 0.6183},
        {"Group": "G10", "Type": "Low-Leak", "Model": "Optimized MS-SIDM (Ours)", "Score": 0.5482},
    ]
    return pd.DataFrame(data)

def plot_risk_scores(df, low_risk_threshold=0.15, medium_risk_threshold=0.4):
    """绘制风险分数柱状图"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 定义颜色方案
    palette = {
        "Traditional Linear": "#95a5a6",
        "Optimized MS-SIDM (Ours)": "#1f77b4"
    }

    # 绘制柱状图
    sns.barplot(
        x="Group", y="Score", hue="Model", data=df,
        palette=palette, edgecolor="black", linewidth=0.8,
        ax=ax, capsize=0.05
    )

    # 添加数值标签 (增大字体)
    for container in ax.containers:
        # fontsize 原 9 -> 12
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=12, rotation=0, fontweight='medium')

    # 添加风险阈值线
    ax.axhline(low_risk_threshold, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.8)
    # fontsize 原 9 -> 12
    ax.text(
        5.6, low_risk_threshold, f"Low Risk Threshold ({low_risk_threshold})",
        color="#27ae60", fontsize=12, fontweight='bold',
        ha='right', va='center',
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=1)
    )

    ax.axhline(medium_risk_threshold, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.8)
    # fontsize 原 9 -> 12
    ax.text(
        5.6, medium_risk_threshold, f"Medium Risk Threshold ({medium_risk_threshold})",
        color="#e67e22", fontsize=12, fontweight='bold',
        ha='right', va='center',
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=1)
    )

    # 添加False Negative标注 (增大字体)
    ax.annotate(
        "False Negative\n(Missed)",
        xy=(2.58, 0.07),
        xytext=(1.3, 0.3),
        arrowprops=dict(
            arrowstyle="->, head_width=0.3",
            color="#c0392b",
            linewidth=1.5,
            connectionstyle="arc3,rad=-0.2"
        ),
        fontsize=13,  # 原 10 -> 13
        color="#c0392b",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9)
    )

    # 添加分组标签和装饰 (增大字体)
    ax.axvline(2.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    # fontsize 原 12 -> 16
    ax.text(1, 0.78, "Safe Scenarios", ha='center', fontsize=16, fontweight='bold', color='#555555')
    ax.text(4, 0.78, "Micro-Leakage Scenarios", ha='center', fontsize=16, fontweight='bold', color='#555555')

    # 设置坐标轴和网格
    ax.set_ylabel("Total Score ($TS$)", fontweight="bold") # 字体大小继承自全局 axes.labelsize (16)
    ax.set_xlabel("")
    ax.set_ylim(0, 0.9)
    sns.despine()
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(False)

    # 设置图例
    # 增加 fontsize=14
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=14)

    return fig, ax

def save_and_show_plot(fig, filename_prefix):
    """保存并显示图表"""
    plt.tight_layout()
    fig.savefig(f"{filename_prefix}.png", dpi=300)
    fig.savefig(f"{filename_prefix}.pdf", dpi=600)
    plt.show()

if __name__ == "__main__":
    set_sci_plotting_style()
    risk_df = create_risk_score_dataframe()
    fig, ax = plot_risk_scores(risk_df)
    save_and_show_plot(fig, "Experiment1_Comparison_LargeFont")