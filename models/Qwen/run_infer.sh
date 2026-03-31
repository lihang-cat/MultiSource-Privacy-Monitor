#!/bin/bash

echo "========================================="
echo "      Qwen 敏感信息检测模型 - 推理模式"
echo "========================================="

# 1. 自动列出 output 目录下的最近模型
echo "🔍 正在扫描 output 目录..."
ls -dt output/*/*/checkpoint-* | head -n 5
echo "-----------------------------------------"

# 2. 让用户输入要加载的 Checkpoint 路径
read -p "请输入要加载的 checkpoint 完整路径 (复制上面的路径): " CKPT_PATH

if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ 错误: 路径不存在 -> $CKPT_PATH"
    exit 1
fi
./
echo "🚀 正在加载模型，进入交互式控制台..."
echo "💡 提示: 输入 'exit' 退出"

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir "$CKPT_PATH" \
