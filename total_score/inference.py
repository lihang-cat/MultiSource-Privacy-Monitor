import json
import torch
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置区域 ==============
MODEL_PATH = "/root/autodl-tmp/MultiSource-contrastive experiment/models/Qwen/output/qwen2.5-7b-best-seed-42/v0-20260329-114544/checkpoint-813-merged"

PARENT_DIR = "/root/autodl-tmp/MultiSource-contrastive experiment/multi-source-scenario-data/test_data/data_single_point"  #data_single_point

# 设置结果输出目录
OUTPUT_DIR = "inference_results_all_v2"

# ⚡️ 核心参数
BATCH_SIZE = 128
# =====================================

# 0. 准备环境
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录: {OUTPUT_DIR}")

# 1. 加载模型
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

system_prompt = "你是一名安全审计专家。请分析文本，判断其中包含的敏感信息类别（Identity, Location, Credential, Safe）。"
valid_labels = ["Identity", "Location", "Credential", "Safe"]


# --- 核心推理函数
def batch_predict(texts):
    prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": system_prompt},

            {"role": "user", "content": f"分析以下文本的敏感信息类别：\n{text}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False  # 建议开启确定性解码
        )

    input_len = inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_len:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def parse_labels(raw_output):
    detected_labels = []
    lower_output = raw_output.lower()
    for label in valid_labels:
        if label.lower() in lower_output:
            detected_labels.append(label)
    if not detected_labels: return ["Safe"]
    if "Safe" in detected_labels and len(detected_labels) > 1:
        detected_labels.remove("Safe")
    return detected_labels


# ===========================================
# 🚩 主循环逻辑 (遍历父目录下的所有子文件夹)
# ===========================================

# 获取父目录下所有的子文件夹
sub_dirs = [d for d in os.listdir(PARENT_DIR) if os.path.isdir(os.path.join(PARENT_DIR, d))]
sub_dirs.sort()

print(f"🔍 扫描到 {len(sub_dirs)} 个待处理任务组: {sub_dirs}")

# 外层循环：遍历每个文件夹
for sub_dir_name in tqdm(sub_dirs, desc="处理任务组"):

    current_data_dir = os.path.join(PARENT_DIR, sub_dir_name)

    # 自动定义该文件夹对应的输出文件名
    current_output_file = os.path.join(OUTPUT_DIR, f"result_{sub_dir_name}.jsonl")

    # --- 修改逻辑开始 ---
    # 这里不再 continue，而是仅打印提示，后续的 'w' 模式会自动覆盖文件
    if os.path.exists(current_output_file):
        print(f"\n⚠️ 文件已存在，将覆盖写入: {os.path.basename(current_output_file)}")
    else:
        print(f"\n📂 正在处理文件夹: {sub_dir_name} -> 输出至: {os.path.basename(current_output_file)}")
    # --- 修改逻辑结束 ---

    # 获取该文件夹下的所有 jsonl
    jsonl_files = glob.glob(os.path.join(current_data_dir, "*.jsonl"))
    batch_buffer = []

    # 打开输出文件 ('w' 模式会清空原文件内容，实现覆盖)
    with open(current_output_file, 'w', encoding='utf-8') as f_out:

        # 遍历该文件夹下的每个 jsonl
        for file_path in jsonl_files:
            source_id = os.path.splitext(os.path.basename(file_path))[0]

            with open(file_path, 'r', encoding='utf-8') as f_in:
                # 这里的 tqdm 可以设置 leave=False，避免进度条刷屏
                for line in tqdm(f_in, desc=f"Reading {source_id}", leave=False):
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if not text: continue

                        batch_buffer.append({"source": source_id, "text": text})

                        # 执行 Batch 推理
                        if len(batch_buffer) >= BATCH_SIZE:
                            batch_texts = [item['text'] for item in batch_buffer]
                            batch_responses = batch_predict(batch_texts)

                            for item, resp in zip(batch_buffer, batch_responses):
                                result_entry = {
                                    "source": item['source'],
                                    "folder": sub_dir_name,  # 额外记录是哪个组的数据
                                    "text": item['text'],
                                    "raw_output": resp,
                                    "predicted_labels": parse_labels(resp)
                                }
                                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                            batch_buffer = []  # 清空

                    except json.JSONDecodeError:
                        continue

        # 处理该文件夹下剩余的数据 (Flush buffer)
        if batch_buffer:
            batch_texts = [item['text'] for item in batch_buffer]
            batch_responses = batch_predict(batch_texts)
            for item, resp in zip(batch_buffer, batch_responses):
                result_entry = {
                    "source": item['source'],
                    "folder": sub_dir_name,
                    "text": item['text'],
                    "raw_output": resp,
                    "predicted_labels": parse_labels(resp)
                }
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

print(f"\n✅ 所有文件夹处理完成！结果保存在: {OUTPUT_DIR}")