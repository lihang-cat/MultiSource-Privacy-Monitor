#!/bin/bash

PROJECT_ROOT=$(pwd)
DATA_PATH="/root/autodl-tmp/MultiSource-contrastive experiment/models/Qwen/data/processed/train_data.jsonl"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 错误: 找不到训练数据 $DATA_PATH"
    exit 1
fi

# 定义需要跑的 3 个随机种子
SEEDS=(42 1024 2024)

# 使用 for 循环依次执行不同种子的训练
for SEED in "${SEEDS[@]}"; do
    echo "======================================================"
    echo "🚀 开始微调 Baichuan2-7B (当前随机种子 Seed: $SEED) ..."
    echo "======================================================"

    # 动态修改输出目录，确保不同种子的权重分别保存在不同文件夹
    OUTPUT_DIR="$PROJECT_ROOT/output/baichuan2-7b-best-seed-$SEED"

    CUDA_VISIBLE_DEVICES=0 swift sft \
        --model baichuan-inc/Baichuan2-7B-Chat \
        --train_type lora \
        --output_dir "$OUTPUT_DIR" \
        \
        --dataset "$DATA_PATH" \
        \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 6 \
        --learning_rate 5e-5 \
        \
        --lora_rank 16 \
        --lora_alpha 32 \
        \
        --max_length 1024 \
        --logging_steps 10 \
        \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 2 \
        --seed "$SEED"  # <====== 核心改动：在这里传入随机种子

    echo "✅ 种子 $SEED 训练完成！权重已保存至: $OUTPUT_DIR"
done

echo "🎉 Baichuan2-7B 所有 3 个种子的微调任务已全部结束！"