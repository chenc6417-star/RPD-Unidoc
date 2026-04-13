#!/bin/bash

# 指定显卡
export CUDA_VISIBLE_DEVICES=0

# 启动 SWIFT 4.0 全量微调
swift sft \
    --model "/home/hk-project-pai00054/wm9835/qwen35/model_weights/Qwen3.5-0.8B" \
    --dataset "./qwen35_stage1_localization_clean.jsonl" \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --learning_rate 1e-6 \
    --max_length 8192 \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --output_dir ./output_qwen35_expert \
    --system "You are a document analysis expert. For each page, output the correct containing bounding boxes [ymin, xmin, ymax, xmax] and corresponding labels, ocr text and alt text for figures or tables in this page.."