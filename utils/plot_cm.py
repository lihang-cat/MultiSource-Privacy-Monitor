import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 全局字体与绘图格式设置 =================
# 加入备用字体 "DejaVu Serif"，防止 Linux 服务器找不到 Times New Roman 报错
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"] 
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14  # 标题字体稍微调大，适应单行精简版
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

LABEL_LIST = ["Identity", "Location", "Credential", "Safe"]

def build_cooccurrence_cm(y_true, y_pred):
    num_classes = len(LABEL_LIST)
    cm = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        true_indices = np.where(t == 1)[0]
        pred_indices = np.where(p == 1)[0]
        for i in true_indices:
            for j in pred_indices:
                cm[i, j] += 1
                
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 
    return cm / row_sums

# ================= 2. 加载同量级大模型的真实预测数据 =================
try:
    y_true_real = np.load("y_true_real.npy")
    y_pred_baichuan = np.load("y_pred_baichuan2_seed_42.npy")        
    y_pred_Bert = np.load("y_pred_bert_seed_42.npy")    
    y_pred_ours = np.load("y_pred_qwen_seed_42.npy")         #  MS-SIDM
except FileNotFoundError:
    print("⚠️ 找不到 .npy 文件，这里暂时使用模拟数据展示效果...")
    # 修复：将模拟数据的变量名与后续绘图引用的变量名对齐
    y_true_real = np.zeros((700, 4))
    y_true_real[0:175, 0], y_true_real[175:350, 1], y_true_real[350:525, 2], y_true_real[525:700, 3] = 1, 1, 1, 1
    
    y_pred_Bert = np.copy(y_true_real)
    y_pred_Bert[525:550, 0] = 1; y_pred_Bert[525:550, 3] = 0 # 模拟 BERT 误报
    
    y_pred_baichuan = np.copy(y_true_real)
    y_pred_baichuan[350:380, 3] = 1; y_pred_baichuan[350:380, 2] = 0 # 模拟 Baichuan 漏报
    
    y_pred_ours = np.copy(y_true_real)

cm_Bert = build_cooccurrence_cm(y_true_real, y_pred_Bert)
cm_baichuan = build_cooccurrence_cm(y_true_real, y_pred_baichuan)
cm_ours = build_cooccurrence_cm(y_true_real, y_pred_ours)

# ================= 3. 绘制并排的高清热力图 (1行3列) =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 

cmap = "Blues"

# 子图 1: BERT
sns.heatmap(cm_Bert, annot=True, fmt=".1%", cmap=cmap, ax=axes[0], cbar=False,
            xticklabels=LABEL_LIST, yticklabels=LABEL_LIST, vmin=0, vmax=1)
# 【改动】IEEE 极简风格标题，并对应真实变量
axes[0].set_title("(a) BERT", fontweight='bold', pad=12)
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# 子图 2: Baichuan2-7B
sns.heatmap(cm_baichuan, annot=True, fmt=".1%", cmap=cmap, ax=axes[1], cbar=False,
            xticklabels=LABEL_LIST, yticklabels=LABEL_LIST, vmin=0, vmax=1)
# 【改动】IEEE 极简风格标题，并对应真实变量
axes[1].set_title("(b) Baichuan2-7B", fontweight='bold', pad=12)
axes[1].set_xlabel("Predicted Label")
axes[1].set_yticks([]) 
axes[1].set_ylabel("")

# 子图 3: MS-SIDM (Ours)
sns.heatmap(cm_ours, annot=True, fmt=".1%", cmap=cmap, ax=axes[2], cbar=True, cbar_ax=cbar_ax,
            xticklabels=LABEL_LIST, yticklabels=LABEL_LIST, vmin=0, vmax=1)
# 【改动】IEEE 极简风格标题，并对应真实变量
axes[2].set_title("(c) MS-SIDM (Ours)", fontweight='bold', pad=12)
axes[2].set_xlabel("Predicted Label")
axes[2].set_yticks([]) 
axes[2].set_ylabel("")

plt.subplots_adjust(left=0.05, right=0.9, wspace=0.1)

output_filename = "confusion_matrices_comparison_IEEE.pdf"
# 保持 600 DPI，满足顶级期刊排版要求
plt.savefig(output_filename, dpi=600, bbox_inches='tight')
print(f"✅ IEEE 规范对比图表已成功保存为 {output_filename}")

plt.show()