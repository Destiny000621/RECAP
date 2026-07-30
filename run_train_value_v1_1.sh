#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localssd/Sichang/recap

export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home
export OPENPI_VLM_CKPT_DIR=/mnt/localssd/Sichang/recap/vlm_ckpt
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export CUDA_VISIBLE_DEVICES=0,1,2,3

exec uv run python scripts/train_value.py \
    --data_dir /mnt/localssd/Sichang/lerobot_home/Sichang0621/vials_recap_v1_1_v21 \
    --checkpoint_dir checkpoints/value_model/yam_vial_v1.1 \
    --batch_size 64 --num_train_steps 6000 \
    --log_interval 100 --save_interval 2000 --val_interval 0 \
    --load_pretrained \
    --tokenizer_path /mnt/localssd/Sichang/recap/vlm_ckpt/tokenizer.model \
    --wandb_mode offline --wandb_project recap-value --wandb_run_name yam_vial_v1
