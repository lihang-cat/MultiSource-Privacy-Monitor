import json

# 输入文件名
INPUT_FILE = "/root/autodl-tmp/MultiSource-contrastive experiment/data/Ablation_Study_data/train_data_v3.jsonl"
# 输出文件名 (Swift 格式)
OUTPUT_FILE = "/root/autodl-tmp/MultiSource-contrastive experiment/models/Qwen/data/processed/train_data_v3.jsonl"

system_prompt = "你是一名安全审计专家。请分析文本，判断其中包含的敏感信息类别（Identity, Location, Credential, Safe）。"

converted_data = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line)
            raw_text = item['text']
            # 将列表标签转为逗号分隔的字符串，例如 "Identity, Location"
            labels_str = ", ".join(item['labels'])

            # 构造一条对话数据
            entry = {
                "system": system_prompt,
                "query": f"分析以下文本的敏感信息类别：\n{raw_text}",
                "response": labels_str
            }
            converted_data.append(entry)
        except Exception as e:
            print(f"Skipping line: {e}")

# 写入新文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for entry in converted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"转换完成，共 {len(converted_data)} 条数据，保存为 {OUTPUT_FILE}")