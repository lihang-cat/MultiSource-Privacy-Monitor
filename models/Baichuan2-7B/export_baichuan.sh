#!/bin/bash

echo "========================================="
echo "      Baichuan2 模型导出 (Merge LoRA)"
echo "========================================="

# 1. 智能尝试列出最近的 checkpoint
# 针对你当前的目录结构进行搜索
echo "🔍 正在扫描 Baichuan2 的最近存档..."

# 尝试搜索你刚才提供的特定目录结构
TARGET_DIR="models/Baichuan2-7B/output"
if [ -d "$TARGET_DIR" ]; then
    ls -dt "$TARGET_DIR"/*/*/checkpoint-* 2>/dev/null | head -n 5
else
    # 如果找不到特定目录，尝试搜索通用的 output 目录
    ls -dt output/*/*/checkpoint-* 2>/dev/null | head -n 5
fi

echo "-----------------------------------------"
echo "💡 提示：您可以直接复制上方列出的路径，或粘贴日志中的完整路径"
echo "-----------------------------------------"

# 2. 接收输入
read -p "请输入 checkpoint 完整路径: " CKPT_PATH

# 清洗输入：去除可能误复制的引号
CKPT_PATH=$(echo "$CKPT_PATH" | tr -d "'\"")

# 3. 验证路径
if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ 错误: 路径不存在 -> $CKPT_PATH"
    echo "   请检查路径中是否有空格或拼写错误。"
    exit 1
fi

echo "🚀 开始合并权重 (Baichuan2 + LoRA)..."
echo "📂 目标路径: $CKPT_PATH"

# 4. 执行合并
# 注意：Baichuan2 需要 trust_remote_code 才能正确加载 modeling 代码
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "$CKPT_PATH" \
    --merge_lora true \
    --safe_serialization true \


echo "✅ 合并完成！"
echo "融合后的模型位于: $CKPT_PATH/merged"