import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def perform_mcnemar_test():
    print("==================================================")
    print("🔬 正在执行 McNemar 显著性检验 (基于 Exact Match)")
    print("==================================================")
    
    # 1. 加载你的真实预测数据矩阵
    try:
        y_true = np.load("y_true_real.npy")
        y_pred_baseline = np.load("y_pred_bert_seed_42.npy")  # 强基线 (比如 BERT,llama)
        y_pred_ours = np.load("y_pred_qwen_seed_42.npy")      # 主角 (MS-SIDM)
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件。请确保 .npy 文件都在当前目录下！\n详细报错: {e}")
        return

    # 2. 计算 Exact Match (EM) 的布尔数组
    # np.all(axis=1) 意味着：只有当这一行的 4 个标签全部预测正确时，才返回 True
    em_baseline = np.all(y_pred_baseline == y_true, axis=1)
    em_ours = np.all(y_pred_ours == y_true, axis=1)

    # 3. 统计 2x2 列联表 (Contingency Table) 的四个象限
    both_correct = np.sum(em_baseline & em_ours)                 # 俩都对
    both_wrong = np.sum((~em_baseline) & (~em_ours))             # 俩都错
    baseline_wrong_ours_correct = np.sum((~em_baseline) & em_ours) # 基线错，我们对 (决定性优势 b)
    baseline_correct_ours_wrong = np.sum(em_baseline & (~em_ours)) # 基线对，我们错 (决定性劣势 c)

    # 构建 McNemar 需要的 2x2 表格
    # [[都对,  基线错/我们对],
    #  [基线对/我们错, 都错]]
    table = [[both_correct, baseline_wrong_ours_correct],
             [baseline_correct_ours_wrong, both_wrong]]

    # 4. 打印直观的对比表
    print("\n📊 2x2 列联表 (Contingency Table):")
    print(f"                       | MS-SIDM (Ours) 正确 | MS-SIDM (Ours) 错误 |")
    print(f"------------------------------------------------------------------------")
    print(f" Baseline (BERT) 正确  |          {both_correct:<14} |          {baseline_correct_ours_wrong:<14} |")
    print(f" Baseline (BERT) 错误  |          {baseline_wrong_ours_correct:<14} |          {both_wrong:<14} |")
    print(f"------------------------------------------------------------------------")
    
    # 5. 执行 McNemar 检验
    # exact=False 表示使用卡方近似 (适用于样本量较大)，correction=True 表示使用连续性校正
    result = mcnemar(table, exact=False, correction=True)
    
    print("\n🧪 检验结果:")
    print(f" - 卡方统计量 (Chi-squared): {result.statistic:.4f}")
    print(f" - p-value (显著性概率): {result.pvalue:.4e}")
    
    # 6. 生成论文结论话术
    print("\n📜 论文写作结论提取:")
    if result.pvalue < 0.001:
        print("🎉 结论: 你的模型提升具有【极强的统计学显著性】(p < 0.001)！")
        print('可以直接把这句话写进论文：')
        print('"McNemar\'s test confirms that the performance improvement of our MS-SIDM over the strongest baseline is highly statistically significant (p < 0.001)."')
    elif result.pvalue < 0.05:
        print("✅ 结论: 你的模型提升具有【统计学显著性】(p < 0.05)。")
        print('可以直接把这句话写进论文：')
        print('"McNemar\'s test confirms that the performance improvement of our MS-SIDM is statistically significant (p < 0.05)."')
    else:
        print("⚠️ 结论: p-value >= 0.05，在统计学上差异不显著。")

if __name__ == "__main__":
    perform_mcnemar_test()