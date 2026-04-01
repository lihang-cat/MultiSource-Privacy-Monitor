import torch
import json
import os
import re
import time
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as st # 用于计算置信区间

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, multilabel_confusion_matrix

# ================ 🔧 配置区域 ===============


MODEL_PATHS = [
    "/root/autodl-tmp/MultiSource-contrastive experiment/models/Llama 2-7B/output/llama2-7b-best-seed-42/v0-20260329-215021/checkpoint-813-merged",
    "/root/autodl-tmp/MultiSource-contrastive experiment/models/Llama 2-7B/output/llama2-7b-best-seed-1024/v0-20260329-233106/checkpoint-813-merged",
    "/root/autodl-tmp/MultiSource-contrastive experiment/models/Llama 2-7B/output/llama2-7b-best-seed-2024/v0-20260330-122524/checkpoint-813-merged"
]
SEED_NAMES = ["42", "1024", "2024"]
                             # synthetic_test_isolated  synthetic_test_stress_extreme cluener_test ai4privacy_golden_test
# 测试集路径 
TEST_DATA_PATH = "/root/autodl-tmp/MultiSource-contrastive experiment/data/test_data/ai4privacy_golden_test.jsonl"

BATCH_SIZE = 64

# ================= 🧹 核心工具：核弹级标签清洗 =================
def normalize_label(raw_label):
    if not isinstance(raw_label, str): return None
    text = raw_label.strip().lower().lstrip('._-')
    if 'identity' in text: return 'Identity'
    if 'location' in text: return 'Location'
    if 'credential' in text: return 'Credential'
    if 'safe' in text: return 'Safe'
    return None

def batch_clean_labels(label_list_of_lists):
    cleaned_batch = []
    for labels in label_list_of_lists:
        clean_row = set()
        for l in labels:
            norm = normalize_label(l)
            if norm: clean_row.add(norm)
        if not clean_row: clean_row.add('Safe')
        cleaned_batch.append(list(clean_row))
    return cleaned_batch

def parse_model_output(text):
    if not text: return []
    found_raw = []
    valid_keywords = ['Identity', 'Location', 'Credential', 'Safe']
    for key in valid_keywords:
        if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
            found_raw.append(key)
    if not found_raw:
        found_raw = [t.strip() for t in text.split(',')]
    return found_raw

# ================= 🚀 模型加载与推理 (保留 LLaMA 专属模板逻辑) ================
def load_optimized_model(model_path):
    print(f"\n🔄 正在加载 Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    #================= 修复 Llama 2 缺失模板 (至关重要) =================
    if getattr(tokenizer, "chat_template", None) is None:
        print("⚠️ 检测到 chat_template 为空，正在注入 Llama 2 标准模板...")
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = '' %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + content + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )

    tokenizer.padding_side = 'left'
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("⚠️ 检测到 pad_token 为 None，已手动设置为 eos_token")

    print(f"🔄 正在加载模型权重 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def predict_batch(model, tokenizer, texts, batch_size=32):
    system_prompt = "你是一名安全审计专家。请分析文本，判断其中包含的敏感信息类别（Identity, Location, Credential, Safe）。"
    predictions, inference_times = [], []

    all_prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"分析以下文本的敏感信息类别：\n{text}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)

    print(f"🚀 开始批量推理 (Total: {len(texts)}, Batch: {batch_size})...")

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Inferencing"):
        batch_prompts = all_prompts[i: i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(model.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        end_time = time.time()
        inference_times.append(end_time - start_time)

        input_len = inputs.input_ids.shape[1]
        decoded_outputs = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
        predictions.extend(decoded_outputs)

    return predictions, inference_times

def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 错误: 找不到测试文件 {TEST_DATA_PATH}"); return

    print(f"📂 读取测试集: {TEST_DATA_PATH}")
    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): test_data.append(json.loads(line))
    texts = [item['text'] for item in test_data]

    print("🧹 正在深度清洗真实标签...")
    true_labels_raw = [[l for l in (item.get('labels', []) if isinstance(item.get('labels', []), list) else item.get('labels', '').split(','))] for item in test_data]
    y_true_cleaned = batch_clean_labels(true_labels_raw)

    mlb = MultiLabelBinarizer(classes=["Identity", "Location", "Credential", "Safe"])
    y_true_matrix = mlb.fit_transform(y_true_cleaned)
    np.save("y_true_real_llama.npy", y_true_matrix)

    f1_scores, acc_scores, fpr_scores = [], [], []

    # ================= 🚀 循环遍历评估 3 个种子 =================
    for idx, model_path in enumerate(MODEL_PATHS):
        seed = SEED_NAMES[idx]
        print(f"\n==================================================")
        print(f"🌟 开始评估 LLaMA 2 Seed: {seed} ({idx+1}/{len(MODEL_PATHS)})")
        print(f"==================================================")

        model, tokenizer = load_optimized_model(model_path)
        pred_texts, inference_times = predict_batch(model, tokenizer, texts, batch_size=BATCH_SIZE)

        print("🧹 正在清洗预测标签...")
        y_pred_cleaned = batch_clean_labels([parse_model_output(t) for t in pred_texts])
        y_pred_matrix = mlb.transform(y_pred_cleaned)

        np.save(f"y_pred_llama_seed_{seed}.npy", y_pred_matrix)

        micro_f1 = f1_score(y_true_matrix, y_pred_matrix, average='micro', zero_division=0)
        exact_acc = accuracy_score(y_true_matrix, y_pred_matrix)
        
        mcm = multilabel_confusion_matrix(y_true_matrix, y_pred_matrix)
        temp_fprs = []
        for i, cls_name in enumerate(mlb.classes_):
            if cls_name != "Safe":
                tn, fp, fn, tp = mcm[i].ravel()
                temp_fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        macro_fpr = np.mean(temp_fprs)

        print(f"\n🎯 LLaMA 2 Seed {seed} 局部报告: Micro-F1={micro_f1:.4f} | Accuracy={exact_acc:.4f} | FPR={macro_fpr:.4f}")
        
        f1_scores.append(micro_f1)
        acc_scores.append(exact_acc)
        fpr_scores.append(macro_fpr)

        # 保存各个种子的 CSV
        pd.DataFrame({
            "Text": texts, "True_Labels": [", ".join(l) for l in y_true_cleaned],
            "Pred_Labels": [", ".join(l) for l in y_pred_cleaned], "Pred_Raw": pred_texts,
            "Is_Correct": [set(t) == set(p) for t, p in zip(y_true_cleaned, y_pred_cleaned)]
        }).to_csv(f"llama_results_seed_{seed}.csv", index=False, encoding='utf-8-sig')

        print("♻️ 评估完毕，正在彻底清理显存...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ================= 📊 终极统计学报告 =================
    mean_f1 = np.mean(f1_scores) * 100
    std_f1 = np.std(f1_scores, ddof=1) * 100
    ci_lower, ci_upper = st.t.interval(0.95, len(f1_scores)-1, loc=mean_f1, scale=st.sem(np.array(f1_scores)*100))

    mean_acc = np.mean(acc_scores) * 100
    std_acc = np.std(acc_scores, ddof=1) * 100
    
    mean_fpr = np.mean(fpr_scores) * 100
    std_fpr = np.std(fpr_scores, ddof=1) * 100

    print("\n" + "🔥" * 25)
    print("🎯 LLaMA 2 三种子最终统计结果 (直接抄进论文回复信)")
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
    main()