#!/usr/bin/env bash
# LoRA SFT on the PRISM-CoT delivery (the settings used for the loss curve and the reproduction scores).
# Usage: bash train_lora_sft.sh <source_root> <output_dir>      (ms-swift 3.3.x, 1 GPU with >= 26 GB)
set -euo pipefail
ROOT=${1:?source_root}; OUT=${2:?output_dir}
HERE=$(cd "$(dirname "$0")" && pwd)
python "$HERE/to_swift.py" --root "$ROOT" --out "$OUT/prism_cot_swift.jsonl"
MAX_PIXELS=302112 swift sft \
  --model Qwen/Qwen2-VL-7B-Instruct --model_type qwen2_vl \
  --train_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear --freeze_vit true \
  --dataset "$OUT/prism_cot_swift.jsonl" --split_dataset_ratio 0.02 --max_length 2048 \
  --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 --learning_rate 1e-4 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --torch_dtype bfloat16 --gradient_checkpointing true --attn_impl sdpa \
  --logging_steps 5 --eval_steps 50 --save_steps 100 --save_total_limit 3 \
  --dataloader_num_workers 4 --dataset_num_proc 8 --report_to tensorboard --seed 42 \
  --output_dir "$OUT/ckpt"
