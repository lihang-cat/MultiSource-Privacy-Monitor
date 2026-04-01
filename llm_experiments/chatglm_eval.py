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

# ================== 🔧 配置区域 ==================

# ⚠️ 替换为你实际跑出来的 3 个 ChatGLM 种子的 merged 权重绝对路径
MODEL_PATHS = [
    "/root/autodl-tmp/MultiSource-contrastive experiment/models/ChatGLM3-6B/output/chatglm3-6b-best-seed-42/v0-20260329-174019/checkpoint-813-merged",
    "/root/autodl-tmp/MultiSource-contrastive experiment/models/ChatGLM3-6B/output/chatglm3-6b-best-seed-1024/v0-20260329-191552/checkpoint-813-merged",
    "/root/autodl-tmp/MultiSource-contrastive experiment/models/ChatGLM3-6B/output/chatglm3-6b-best-seed-2024/v0-20260329-202802/checkpoint-813-merged"
]
SEED_NAMES = ["42", "1024", "2024"]
# synthetic_test_isolated  synthetic_test_stress_extreme cluener_test ai4privacy_golden_test
TEST_DATA_PATH = "/root/autodl-tmp/MultiSource-contrastive experiment/data/test_data/synthetic_test_isolated.jsonl"

BATCH_SIZE = 32   
MAX_NEW_TOKENS = 32

# ================= 🧹 标签清洗 ==================

def normalize_label(raw_label):
    if not isinstance(raw_label, str):
        return None
    text = raw_label.strip().lower().lstrip('._-')
    if 'identity' in text: return 'Identity'
    if 'location' in text: return 'Location'
    if 'credential' in text: return 'Credential'
    if 'safe' in text: return 'Safe'
    return None

def batch_clean_labels(label_list_of_lists):
    cleaned = []
    for labels in label_list_of_lists:
        row = set()
        for l in labels:
            norm = normalize_label(l)
            if norm:
                row.add(norm)
        if not row:
            row.add('Safe')
        cleaned.append(list(row))
    return cleaned

def parse_model_output(text):
    if not text:
        return []
    keys = ['Identity', 'Location', 'Credential', 'Safe']
    found = []
    for k in keys:
        if re.search(r'\b' + re.escape(k) + r'\b', text, re.IGNORECASE):
            found.append(k)
    if not found:
        found = [t.strip() for t in text.split(',')]
    return found

# ================= 🚑 ChatGLM 安全生成 ==================

@torch.no_grad()
def safe_generate(
    model,
    input_ids,
    attention_mask=None,
    max_new_tokens=32,
    pad_token_id=None,
):
    """
    ChatGLM 专用安全生成 (纯 greedy, 防崩溃)
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    generated = input_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_tokens], dim=-1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )

        if pad_token_id is not None:
            if torch.all(next_tokens.squeeze(-1) == pad_token_id):
                break

    return generated

# ================= 🚀 模型加载 ==================

def load_optimized_model(model_path):
    print(f"\n🔄 正在加载 Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ================= 🚑 补丁 1: 修复 Tokenizer padding_side 报错 =================
    if "ChatGLM" in tokenizer.__class__.__name__:
        print("🚑 [Patch 1] 正在注入 padding_side 兼容性补丁...")
        _original_pad = tokenizer._pad

        def _new_pad(self, *args, **kwargs):
            if 'padding_side' in kwargs:
                kwargs.pop('padding_side')
            return _original_pad(*args, **kwargs)

        import types
        tokenizer._pad = types.MethodType(_new_pad, tokenizer)

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 2  # ChatGLM eos_id
            tokenizer.pad_token = "</s>"

    print(f"🔄 正在加载模型权重 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # ================= 🚑 补丁 2: 修复 ChatGLMConfig num_hidden_layers =================
    if not hasattr(model.config, "num_hidden_layers"):
        print("🚑 [Patch 2] 检测到 num_hidden_layers 缺失，正在注入属性映射...")
        if hasattr(model.config, "num_layers"):
            model.config.num_hidden_layers = model.config.num_layers
        elif hasattr(model.config, "num_hidden_layers"):
            pass
        else:
            print("⚠️ 警告：强制设置层数为 28 (ChatGLM3-6B)")
            model.config.num_hidden_layers = 28

    model.eval()
    return model, tokenizer

# ================= 🔮 批量推理 ==================

def predict_batch(model, tokenizer, texts, batch_size=16):
    system_prompt = "你是一名安全审计专家。请分析文本，判断其中包含的敏感信息类别（Identity, Location, Credential, Safe）。"
    predictions, inference_times = [], []

    all_prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"分析以下文本的敏感信息类别：\n{text}"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_prompts.append(prompt)

    print(f"🚀 开始推理: {len(texts)} 条，batch={batch_size}")

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Inferencing"):
        batch_prompts = all_prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(model.device)

        start = time.time()
        generated_ids = safe_generate(
            model=model,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
        end = time.time()
        inference_times.append(end - start)

        input_len = inputs.input_ids.shape[1]
        gen_only = generated_ids[:, input_len:]

        outputs = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        
        clean_outputs = []
        for out in outputs:
            if not out:
                clean_outputs.append("")
                continue
            first_line = out.strip().splitlines()[0]
            if "<|user|>" in first_line:
                first_line = first_line.split("<|user|>")[0].strip()
            clean_outputs.append(first_line)

        predictions.extend(clean_outputs)

    return predictions, inference_times

# ================= 🧪 主流程 ==================

def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 找不到测试文件 {TEST_DATA_PATH}"); return

    print(f"📂 读取测试集: {TEST_DATA_PATH}")
    test_data = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): test_data.append(json.loads(line))

    texts = [x["text"] for x in test_data]

    print("🧹 清洗真实标签")
    true_raw = []
    for item in test_data:
        lbl = item.get("labels", [])
        if isinstance(lbl, str): true_raw.append(lbl.split(","))
        elif isinstance(lbl, list): true_raw.append(lbl)
        else: true_raw.append([])

    y_true = batch_clean_labels(true_raw)

    mlb = MultiLabelBinarizer(classes=["Identity", "Location", "Credential", "Safe"])
    y_true_m = mlb.fit_transform(y_true)
    np.save("y_true_real_chatglm.npy", y_true_m)

    f1_scores, acc_scores, fpr_scores = [], [], []

    # ================= 🚀 循环遍历评估 3 个种子 =================
    for idx, model_path in enumerate(MODEL_PATHS):
        seed = SEED_NAMES[idx]
        print(f"\n==================================================")
        print(f"🌟 开始评估 ChatGLM3 Seed: {seed} ({idx+1}/{len(MODEL_PATHS)})")
        print(f"==================================================")

        model, tokenizer = load_optimized_model(model_path)
        preds, times = predict_batch(model, tokenizer, texts, batch_size=BATCH_SIZE)

        print("🧹 清洗预测标签")
        pred_raw = [parse_model_output(t) for t in preds]
        y_pred = batch_clean_labels(pred_raw)
        y_pred_m = mlb.transform(y_pred)
        
        np.save(f"y_pred_chatglm_seed_{seed}.npy", y_pred_m)

        micro_f1 = f1_score(y_true_m, y_pred_m, average="micro", zero_division=0)
        exact_acc = accuracy_score(y_true_m, y_pred_m)

        mcm = multilabel_confusion_matrix(y_true_m, y_pred_m)
        temp_fprs = []
        for i, cls_name in enumerate(mlb.classes_):
            if cls_name != "Safe":
                tn, fp, fn, tp = mcm[i].ravel()
                temp_fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        macro_fpr = np.mean(temp_fprs)

        print(f"\n🎯 ChatGLM3 Seed {seed} 局部报告: Micro-F1={micro_f1:.4f} | Accuracy={exact_acc:.4f} | FPR={macro_fpr:.4f}")

        f1_scores.append(micro_f1)
        acc_scores.append(exact_acc)
        fpr_scores.append(macro_fpr)

        pd.DataFrame({
            "Text": texts, "True_Labels": [", ".join(x) for x in y_true],
            "Pred_Labels": [", ".join(x) for x in y_pred], "Pred_Raw": preds,
            "Exact_Match": [set(t) == set(p) for t, p in zip(y_true, y_pred)]
        }).to_csv(f"ChatGLM3_results_seed_{seed}.csv", index=False, encoding="utf-8-sig")

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
    print("🎯 ChatGLM3 三种子最终统计结果 (直接抄进论文回复信)")
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