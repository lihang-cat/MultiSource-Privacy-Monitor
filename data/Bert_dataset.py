import json
import torch
from torch.utils.data import Dataset

# ================= 配置区 (修改点 1) =================
# 将 "Safe" 加入列表，现在它是第 4 个分类
# 顺序建议：把 Safe 放在最后，方便查看
LABEL_LIST = ["Identity", "Credential", "Location", "Safe"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)  # 现在等于 4


class PrivacyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"Checking data in {data_path}...")

        # 读取数据逻辑 (保持你喜欢的健壮性检查不变)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)

                    # --- 数据清洗 ---
                    raw_text = item.get('text', '')
                    if raw_text is None:
                        text_str = ""
                    else:
                        text_str = str(raw_text).strip()

                    if len(text_str) < 1:
                        continue

                    item['text'] = text_str

                    # --- 预处理标签列表 ---
                    # 某些数据集可能标签为空列表 []，这通常意味着 Safe
                    # 如果你的数据里明确写了 ["Safe"]，这一步是双重保险
                    labels = item.get('labels', [])
                    if not labels:
                        labels = ["Safe"]
                    item['labels'] = labels

                    self.data.append(item)

                except json.JSONDecodeError:
                    print(f"Error parsing JSON at line {line_num + 1}")
                    continue

        print(f"Loaded {len(self.data)} valid samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['text']
        labels = item['labels']

        # 1. BERT 编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 2. 标签转 One-Hot 向量 (修改点 2)
        # 初始化全 0 向量: [0, 0, 0, 0] (长度为4)
        label_tensor = torch.zeros(NUM_LABELS, dtype=torch.float)

        if isinstance(labels, list):
            for l in labels:
                # === 核心修改：不再跳过 Safe ===
                # 只要标签在我们的列表里，就给对应位置标 1
                if l in LABEL2ID:
                    idx = LABEL2ID[l]
                    label_tensor[idx] = 1.0

        # 此时：
        # Identity -> [1, 0, 0, 0]
        # Safe     -> [0, 0, 0, 1]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }