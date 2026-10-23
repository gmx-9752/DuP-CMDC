set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

OUT="${TRITSP_OUTPUT_DIR:-./outputs}"
RESUME_ARGS=()
if [[ -n "${TRITSP_RESUME_CKPT:-}" ]]; then
  RESUME_ARGS+=(--resume_ckpt "${TRITSP_RESUME_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${TRITSP_GPU_IDS:-2,3}" torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --config "${TRITSP_CONFIG:-configs/SanaDiT-600M-1024-mydata-greedy.yaml}" \
  --rgb_list "${TRITSP_RGB_TRAIN_LIST:-./data/train/rgb_train.txt}" \
  --thermal_list "${TRITSP_THERMAL_TRAIN_LIST:-./data/train/ir_train.txt}" \
  --prompt_list "${TRITSP_PROMPT_TRAIN_LIST:-./data/train/prompt.list}" \
  --val_rgb_list "${TRITSP_RGB_VAL_LIST:-./data/val/rgb_val.txt}" \
  --val_thermal_list "${TRITSP_THERMAL_VAL_LIST:-./data/val/ir_val.txt}" \
  --val_prompt_list "${TRITSP_PROMPT_VAL_LIST:-./data/val/prompt.list}" \
  --output_dir "$OUT" \
  --preprocess_mode pad64 \
  --stop_timestep 45 \
  --high_boundary 300 \
  --middle_boundary 85 \
  --epochs "${TRITSP_EPOCHS:-40}" \
  --lr_schedule_epochs "${TRITSP_LR_SCHEDULE_EPOCHS:-40}" \
  --lr 5e-5 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --warmup_steps 500 \
  --grad_clip 1.0 \
  --stage_cycle "${TRITSP_STAGE_CYCLE:-high,middle,high,middle,low}" \
  --lambda_path 1.0 \
  --lambda_dist 0.10 \
  --lambda_bridge 0.20 \
  --lambda_spec 0.0 \
  --lambda_patch 0.03 \
  --lambda_reg 1e-4 \
  --semantic_negative_prob 0.0 \
  --patch_quality_probability 0.10 \
  --patch_crop_latent_size 32 \
  --patch_edge_weight 0.20 \
  --patch_lpips_weight 0.05 \
  --use_patch_lpips \
  --patch_val_pairs 0 \
  --patch_val_every 4 \
  --patch_val_crop_latent_size 32 \
  --patch_val_lpips \
  --patch_val_lpips_weight 0.50 \
  --val_every 2 \
  --val_max_pairs 0 \
  --val_edges_per_stage 1 \
  --save_every 2 \
  --tritsp_cond_dim 512 \
  --tritsp_num_heads 8 \
  --tritsp_condition_tokens 12 \
  --tritsp_tsp_condition_tokens 8 \
  --tritsp_spp_condition_tokens 4 \
  --tritsp_condition_decoder_layers 2 \
  --tritsp_delta_rank 64 \
  --tritsp_gate_max 0.5 \
  --tritsp_inject_block_indices 9,14,18,23 \
  --lowrank_warmup_epochs "${TRITSP_LOWRANK_WARMUP_EPOCHS:-4}" \
  --branch_log_every "${TRITSP_BRANCH_LOG_EVERY:-200}" \
  "${RESUME_ARGS[@]}"
