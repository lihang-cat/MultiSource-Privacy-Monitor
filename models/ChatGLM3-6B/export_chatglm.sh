#!/bin/bash

echo "========================================="
echo "      ChatGLM3-6B 模型导出 (Merge LoRA)"
echo "========================================="

# 1. 智能查找
# 对应您训练脚本中的 OUTPUT_DIR="$PROJECT_ROOT/output/chatglm3-6b-lora"

BASE_DIR="/root/autodl-tmp/MultiSource-contrastive experiment/output/chatglm3-6b-lora"

echo "🔍 正在扫描 output 目录下的最近存档..."
if [ -d "$BASE_DIR" ]; then
    ls -dt "$BASE_DIR"/*/*/checkpoint-* 2>/dev/null | head -n 5
else
    # 如果找不到特定目录，尝试全盘搜索 output
    ls -dt output/*/*/checkpoint-* 2>/dev/null | head -n 5
fi

echo "-----------------------------------------"
echo "💡 提示：请复制上方列出的完整路径 (包含 checkpoint-xxx)"
echo "-----------------------------------------"

# 2. 接收输入
read -p "请输入 checkpoint 完整路径: " CKPT_PATH

# === 🛡️ 核心防错处理 ===
# 1. 去除引号 (防止用户输入 "/path/to/file")
CKPT_PATH=$(echo "$CKPT_PATH" | tr -d "'\"")
# 2. 去除首尾空格 (这是导致 HFValidationError 的元凶)
CKPT_PATH=$(echo "$CKPT_PATH" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
# ======================

# 3. 验证路径
if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ 错误: 路径不存在 -> [$CKPT_PATH]"
    echo "   请检查路径是否正确，或者是否包含未转义的特殊字符。"
    exit 1
fi

echo "🚀 开始合并权重 (ChatGLM3 + LoRA)..."
echo "📂 源路径: $CKPT_PATH"

# 4. 执行合并
# 注意：Swift Export 不需要 --trust_remote_code 参数，它会自动读取 adapter_config.json
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "$CKPT_PATH" \
    --merge_lora true \
    --safe_serialization true

echo "✅ 合并完成！"
echo "融合后的模型位于: $CKPT_PATH-merged"