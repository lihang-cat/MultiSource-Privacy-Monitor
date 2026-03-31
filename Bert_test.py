import torch
import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import gc
import numpy as np
import pandas as pd
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import scipy.stats as st # 新增：用于计算置信区间

from sklearn.metrics import classification_report, accuracy_score, f1_score, multilabel_confusion_matrix

# --- 导入你的自定义模块 ---
from data.Bert_dataset import PrivacyDataset, NUM_LABELS, LABEL_LIST
from models.Bert import BertForSentenceClassification

# ================= 配置区 =================
CONFIG = {
    # 1. 关键路径                  # synthetic_test_isolated  synthetic_test_stress_extreme cluener_test ai4privacy_golden_test
    "test_data_path": "/root/autodl-tmp/MultiSource-contrastive experiment/data/test_data/synthetic_test_stress_extreme.jsonl",
    
    # ⚠️ 替换为你实际的 3 个不同种子的 BERT 权重路径  Roberta-1024
    "model_paths": [
        "./checkpoints/Bert-42/model_epoch_10.pth",
        # "./checkpoints/Roberta-1024/model_epoch_10.pth",
        # "./checkpoints/Roberta-2024/model_epoch_10.pth"
    ],
    "seed_names": ["42", "1024", "2024"],
    
    "base_model": "bert-base-chinese",  # bert-base-chinese 或  hfl/chinese-roberta-wwm-ext-large

    # 2. 推理参数
    "batch_size": 64,
    "max_len": 128,
    "threshold": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def test_model():
    print(f"=== Starting Final 3-Seed Evaluation on Test Set ===")
    print(f"Device: {CONFIG['device']}")

    # ================= 1. 公共数据加载 (只执行一次) =================
    print(f"\n🔄 加载全局 Tokenizer: {CONFIG['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])

    if not os.path.exists(CONFIG['test_data_path']):
        print(f"❌ Error: Test file not found at {CONFIG['test_data_path']}")
        return

    print(f"📂 Loading test data from {CONFIG['test_data_path']}...")
    test_dataset = PrivacyDataset(CONFIG['test_data_path'], tokenizer, max_len=CONFIG['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    print(f"✅ Test samples: {len(test_dataset)}")

    # 准备记录分数的列表
    f1_scores = []
    acc_scores = []
    fpr_scores = []

    # ================= 2. 循环遍历评估 3 个种子 =================
    for idx, model_path in enumerate(CONFIG['model_paths']):
        seed = CONFIG['seed_names'][idx]
        print("\n" + "="*50)
        print(f"🚀 开始评估 BERT Seed: {seed} ({idx+1}/{len(CONFIG['model_paths'])})")
        print("="*50)

        if not os.path.exists(model_path):
            print(f"❌ Error: Model weights not found at {model_path}")
            continue

        # 3. 初始化并加载当前种子的权重
        model = BertForSentenceClassification(CONFIG['base_model'], num_labels=NUM_LABELS)
        state_dict = torch.load(model_path, map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
        model.to(CONFIG['device'])
        model.eval()

        # 4. 推理循环
        y_true, y_pred, inference_times = [], [], []

        print("🔮 Running inference...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(CONFIG['device'])
                mask = batch['attention_mask'].to(CONFIG['device'])
                labels = batch['labels'].to(CONFIG['device'])

                start_time = time.time()
                probs = model(input_ids, mask)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                preds = (probs > CONFIG['threshold']).float()

                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        # 5. 拼接计算
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        # 保存全局 y_true (只需保存一次) 和 当前种子的 y_pred
        if idx == 0:
            np.save("y_true_real_bert.npy", y_true)
        np.save(f"y_pred_bert_seed_{seed}.npy", y_pred)

        # 6. 计算局部指标
        report = classification_report(y_true, y_pred, target_names=LABEL_LIST, digits=4, zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        exact_accuracy = accuracy_score(y_true, y_pred)

        # 计算 FPR
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        fpr_dict = {}
        for i, cls_name in enumerate(LABEL_LIST):
            tn, fp, fn, tp = mcm[i].ravel()
            fpr_dict[cls_name] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        sensitive_classes = [c for c in LABEL_LIST if c != "Safe"]
        sensitive_fprs = [fpr_dict[c] for c in sensitive_classes]
        macro_sensitive_fpr = np.mean(sensitive_fprs) if sensitive_fprs else 0.0

        print(f"🎯 BERT Seed {seed} 局部报告: Micro-F1={micro_f1:.4f} | Accuracy={exact_accuracy:.4f} | FPR={macro_sensitive_fpr:.4f}")

        f1_scores.append(micro_f1)
        acc_scores.append(exact_accuracy)
        fpr_scores.append(macro_sensitive_fpr)

        # 清理内存
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ================= 3. 终极统计学报告 =================
    if not f1_scores:
        print("❌ 所有模型均加载失败，无法进行统计。")
        return

    mean_f1 = np.mean(f1_scores) * 100
    std_f1 = np.std(f1_scores, ddof=1) * 100
    ci_lower, ci_upper = st.t.interval(0.95, len(f1_scores)-1, loc=mean_f1, scale=st.sem(np.array(f1_scores)*100))

    mean_acc = np.mean(acc_scores) * 100
    std_acc = np.std(acc_scores, ddof=1) * 100
    
    mean_fpr = np.mean(fpr_scores) * 100
    std_fpr = np.std(fpr_scores, ddof=1) * 100

    print("\n" + "🔥" * 25)
    print("🎯 Roberta 三种子最终统计结果 (直接抄进论文回复信)")
    print("🔥" * 25)
    print(f"Micro-F1 分数:    {mean_f1:.2f} ± {std_f1:.2f} %")
    print(f"👉 95% 置信区间:  [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print("-" * 50)
    print(f"Exact Accuracy:   {mean_acc:.2f} ± {std_acc:.2f} %")
    print(f"Macro FPR (误报): {mean_fpr:.2f} ± {std_fpr:.2f} %")
    print("-" * 50)
    print(f"原始 F1 记录:     {[round(x*100, 2) for x in f1_scores]}")
    print("=" * 50)


if __name__ == "__main__":
    test_model()