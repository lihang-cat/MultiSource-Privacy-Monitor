import pandas as pd
import json
import math
import os
import glob

# ================= 配置区域 =================
# 1. 输入目录
INPUT_DIR = "inference_results_all_v1"

# 2. 输出报告文件名
SUMMARY_FILE = "threat_analysis_exp3_fixed_env.csv"

# 3. 基础信誉分
DEFAULT_REP_SCORE = 0.6

# 4. 固定总源域数量
FIXED_TOTAL_SOURCES = 4
# 5 密度补偿因子
Density_alpha = 0.8

class MultiTypeThreatModel:
    def __init__(self, density_alpha=Density_alpha):
        """
        初始化
        :param density_alpha: 密度补偿系数 (0.8)
        """
        self.weights = {
            "IDENTITY": 0.8,
            "LOCATION": 0.7,
            "CREDENTIAL": 1.0,
            "OTHER": 0.5
        }
        self.base_val = 0.2
        self.coeff_val = 0.8
        self.density_alpha = density_alpha
        # 环境因子权重配置
        self.w_env = {'base': 0.3, 'dynamic': 0.5, 'rep': 0.2}

    def calculate_score(self, counts, total_records, compromised_sources, total_sources, rep_score):
        """
        计算总分
        :param total_sources: 这里传入的将是固定的 FIXED_TOTAL_SOURCES (4)
        """
        # --- 步骤 1: 分类型计算独立风险 ---
        type_risks = []
        active_types = 0

        print(f"--- [计算详情] 数据量:{total_records} | 泄露源:{compromised_sources}/{total_sources} ---")

        for category, count in counts.items():
            if count <= 0: continue
            active_types += 1

            w_i = self.weights.get(category, self.weights["OTHER"])
            r_i = min(count / total_records, 1.0) if total_records > 0 else 0.0
            m_i = self.base_val + self.coeff_val * math.sqrt(r_i)


            # 密度补偿
            d_i = 1.0 + (self.density_alpha * r_i)

            
            raw_risk = w_i * m_i * d_i

            risk_i = min(raw_risk, 1.0)

            type_risks.append(raw_risk)


        # --- 步骤 2: 聚合内容风险 (S_content) ---
        if not type_risks:
            s_content = 0.0
        else:
            non_risk_prob = 1.0
            for r in type_risks:
                non_risk_prob *= (1.0 - r)
            s_content = 1.0 - non_risk_prob

        # --- 步骤 3: 计算环境放大因子 (E_env) ---
        # [核心修改点生效处]：这里的 total_sources 是固定的 4
        # 如果 compromised_sources 是 2，则 r_source = 0.5 (而不是 1.0)
        r_source = compromised_sources / total_sources if total_sources > 0 else 0

        # 防止数据异常导致分子大于分母
        r_source = min(r_source, 1.0)

        # 信息深度
        r_coverage = min(1.0, active_types / 3.0)

        # 动态最大化
        r_dynamic = max(r_source, r_coverage)
        p_rep = 1.0 - rep_score

        e_env = (self.w_env['base'] +
                 self.w_env['dynamic'] * r_dynamic +
                 self.w_env['rep'] * p_rep)

        print(f"  ==> 内容分 S_content: {s_content:.4f}")
        print(
            f"  ==> 环境分 E_env: {e_env:.4f} (源广度 R_source: {compromised_sources}/{total_sources}={r_source:.2f})")

        # --- 步骤 4: 最终得分 ---
        final_score = min(s_content * e_env, 1.0)
        print(f"  ==> ✅ 最终评分 TS: {final_score:.4f}")

        return round(final_score, 4), r_source  # 返回分值和源广度供记录


def get_risk_assessment(final_score):
    if final_score >= 0.9:
        return "CRITICAL", "🔥"
    elif final_score >= 0.7:
        return "HIGH", "🔴"
    elif final_score >= 0.4:
        return "MEDIUM", "🟡"
    elif final_score >= 0.15:
        return "LOW", "🔵"
    else:
        return "SAFE", "✅"


def is_sensitive(labels):
    set_labels = set(labels)
    if "Safe" in set_labels and len(set_labels) == 1: return False
    return True


# ================= 主程序 =================

files = glob.glob(os.path.join(INPUT_DIR, "result_*.jsonl"))
files.sort()
summary_data = []

print(f"🚀 开始执行实验三：环境态势分析 (Total Sources Fixed = {FIXED_TOTAL_SOURCES})\n")

for file_path in files:
    file_name = os.path.basename(file_path)
    group_name = file_name.replace("result_", "").replace(".jsonl", "")

    # 简单的文件过滤，只跑本次实验相关的组 (可选)
    # if "safe" in group_name: continue

    # 读取数据
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except:
        continue

    if not data: continue
    df = pd.DataFrame(data)
    df['is_sensitive_record'] = df['predicted_labels'].apply(is_sensitive)

    # 1. 统计标签
    df_exploded = df.explode('predicted_labels')
    tag_counts = df_exploded[df_exploded['predicted_labels'] != 'Safe']['predicted_labels'].value_counts()

    counts = {
        "IDENTITY": tag_counts.get("Identity", 0),
        "LOCATION": tag_counts.get("Location", 0),
        "CREDENTIAL": tag_counts.get("Credential", 0)
    }

    # 2. 统计受污染的来源数
    # 逻辑：找出所有含有敏感数据的 source
    source_risk_status = df.groupby('source')['is_sensitive_record'].any()
    risky_sources_count = source_risk_status.sum()  # 例如：g2_2src 只有2个源受污染，这里就是2

    # 注意：这里的 total_records 依然是当前文件的数据量
    total_records = len(df)

    # --- 🔥 调用模型 ---
    model = MultiTypeThreatModel()

    # [关键修改]：传入 FIXED_TOTAL_SOURCES
    final_score, r_src_val = model.calculate_score(
        counts=counts,
        total_records=total_records,
        compromised_sources=risky_sources_count,
        total_sources=FIXED_TOTAL_SOURCES,  # <--- 强制设为 4
        rep_score=DEFAULT_REP_SCORE
    )

    level_code, level_icon = get_risk_assessment(final_score)

    summary_data.append({
        "Group": group_name,
        "Score": final_score,
        "Level": level_code,
        "R_Source": r_src_val,  # 记录这一项用于画图
        "Compromised/Total": f"{risky_sources_count}/{FIXED_TOTAL_SOURCES}",
        "Sources": risky_sources_count
    })

# ================= 输出 =================
if summary_data:
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(by="Score", ascending=False)

    print("\n" + "=" * 50)
    print(f"📊 实验三结果 (环境分母统一为 {FIXED_TOTAL_SOURCES})")
    print("=" * 50)
    print(df_summary[["Group", "Score", "Level", "Compromised/Total"]].to_string(index=False))

    df_summary.to_csv(SUMMARY_FILE, index=False)
    print(f"\n结果已保存至 {SUMMARY_FILE}，可用于绘制 Fig 4。")