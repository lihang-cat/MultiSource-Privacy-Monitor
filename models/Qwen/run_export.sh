#!/bin/bash

echo "========================================="
echo "      Qwen 模型导出 (Merge LoRA)"
echo "========================================="

# 1. 列出最近的 checkpoint
ls -dt output/*/*/checkpoint-* | head -n 5
echo "-----------------------------------------"

read -p "请输入要合并的 checkpoint 完整路径: " CKPT_PATH

if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ 错误: 路径不存在"
    exit 1
fi

echo "🚀 开始合并权重..."
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "$CKPT_PATH" \
    --merge_lora true

echo "✅ 合并完成！"
echo "融合后的模型位于该 checkpoint 目录下的 'merged' 文件夹中。"