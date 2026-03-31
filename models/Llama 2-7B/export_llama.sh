#!/bin/bash
export MODELSCOPE_CACHE="/root/autodl-tmp/modelscope/hub" 
echo "========================================="
echo "      Llama 2 模型导出 (Merge LoRA)"
echo "========================================="

# 1. 列出最近的 checkpoint
# 这里的路径匹配规则适配 swift 的输出结构: output/实验名/版本号/checkpoint-xxx
echo "🔍 正在扫描 output 目录下的最近存档..."
ls -dt output/*/*/checkpoint-* 2>/dev/null | head -n 5

if [ $? -ne 0 ]; then
    echo "⚠️  警告: 在当前 output 目录下没找到 checkpoint。"
    echo "    请确认您是在项目根目录下运行此脚本。"
fi
echo "-----------------------------------------"

# 2. 接收输入
read -p "请输入要合并的 checkpoint 完整路径: " CKPT_PATH

# 去除用户可能意外输入的引号 (为了脚本健壮性)
CKPT_PATH=$(echo "$CKPT_PATH" | tr -d "'\"")

# 3. 验证路径
if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ 错误: 路径不存在 -> $CKPT_PATH"
    exit 1
fi

echo "🚀 开始合并权重 (Llama 2 + LoRA)..."
echo "📂 目标路径: $CKPT_PATH"

# 4. 执行合并
# 注意："$CKPT_PATH" 必须加引号，因为你的路径里有空格
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "$CKPT_PATH" \
    --merge_lora true \
    --safe_serialization true

echo "✅ 合并完成！"
echo "融合后的模型位于该 checkpoint 目录下的 'merged' 文件夹中。"
echo "您可以在推理时将 MODEL_PATH 指向该 merged 目录。"