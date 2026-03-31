import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer,  get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np


# --- 导入自定义模块 ---

from data.Bert_dataset import PrivacyDataset, NUM_LABELS, LABEL_LIST
from models.Bert import BertForSentenceClassification



# ================= 配置参数 (Hyperparameters) =================
CONFIG = {
    # 1. 路径配置
    "train_path": "data/train_data/synthetic_train_total.jsonl",  # 独立的训练集文件
    "output_dir": "./checkpoints/Bert-2024",  # 模型保存路径

    # 2. 模型配置
    "model_name": "bert-base-chinese",  # bert-base-chinese 或  hfl/chinese-roberta-wwm-ext-large
    "max_len": 128,

    # 3. 训练超参
    "batch_size": 24,
    "epochs": 10,
    "learning_rate": 2e-5,  # 微调黄金学习率
    "warmup_ratio": 0.1,  # 预热比例
    "grad_clip": 1.0,  # 梯度裁剪
    "seed": 42,  # 固定随机种子
    "threshold": 0.5  # 判定阈值 (Sigmoid > 0.5)
}

#  SEEDS=(42 1024 2024)
# ================= 辅助函数：固定随机种子 =================
def set_seed(seed_value=2024):
    """确保实验结果可复现 (Reproducibility)"""

    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


# ================= 核心：评估函数 (保留供测试集使用) =================
def evaluate(model, dataloader, device, threshold=0.5):
    """
    自适应评估逻辑：计算 Micro-F1 和详细分类报告
    """
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (返回 loss)
            loss = model(input_ids, mask, labels=labels)
            total_loss += loss.item()

            # 获取预测概率 (Sigmoid)
            # model.forward 在没有 labels 时返回概率
            probs = model(input_ids, mask)

            # 将概率转换为 0/1 (Hard Thresholding)
            preds = (probs > threshold).float()

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    # 拼接所有 batch 的结果
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # 计算平均 Loss
    avg_loss = total_loss / len(dataloader)

    # 计算 Micro-F1 (多标签分类中最关键的指标)
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # 生成详细报告 (包含每个类别的 Precision/Recall)
    # target_names 对应 ["Identity", "Credential", "Location"]
    report = classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0)

    return avg_loss, f1_micro, report


# ================= 主训练循环 =================
def train():
    # 1. 初始化环境
    set_seed(CONFIG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    # 2. 加载数据
    print(f"Loading Tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    print(f"Loading Training Set from {CONFIG['train_path']}...")
    train_dataset = PrivacyDataset(CONFIG['train_path'], tokenizer, max_len=CONFIG['max_len'])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    print(f"Dataset Ready: {len(train_dataset)} Train Samples")

    # 3. 初始化模型
    model = BertForSentenceClassification(CONFIG['model_name'], num_labels=NUM_LABELS)
    model.to(device)

    # 4. 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    total_steps = len(train_loader) * CONFIG['epochs']
    num_warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # 5. 训练 Loop
    print("\n--- Starting Training (No Validation) ---")
    for epoch in range(CONFIG['epochs']):
        # === Training Phase ===
        model.train()
        train_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()

            # Forward
            loss = model(input_ids, mask, labels=labels)

            # Backward
            loss.backward()

            # Gradient Clipping (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # 简单的进度打印
            if step % 50 == 0 and step > 0:
                print(f"  Epoch {epoch + 1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']} Completed. Average Train Loss: {avg_train_loss:.4f}")

        if epoch + 1 == CONFIG['epochs']:
            save_path = os.path.join(CONFIG['output_dir'], f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Model Saved to {save_path}!")

    print("\nTraining Completed! All models saved at:", CONFIG['output_dir'])


if __name__ == "__main__":
    train()