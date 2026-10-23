set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ROOT="${TRITSP_MODEL_DIR:-./outputs}"
BRANCH_ARGS=()
if [[ "${TRITSP_BRANCH_LOG_EVAL:-1}" == "1" ]]; then
  BRANCH_ARGS+=(--branch_contribution_log)
fi

CUDA_VISIBLE_DEVICES="${TRITSP_GPU_IDS:-2,3}" torchrun \
  --standalone \
  --nproc_per_node=2 \
  evaluate.py \
  --config "${TRITSP_CONFIG:-configs/SanaDiT-600M-1024-mydata-greedy.yaml}" \
  --checkpoint "${TRITSP_CHECKPOINT:-${ROOT}/best_val_macro.pt}" \
  --text_cache "${TRITSP_TEXT_CACHE:-${ROOT}/text_cache.pt}" \
  --rgb_list "${TRITSP_RGB_TEST_LIST:-./data/test/test_rgb.list}" \
  --thermal_list "${TRITSP_THERMAL_TEST_LIST:-./data/test/test_ir.list}" \
  --prompt_list "${TRITSP_PROMPT_TEST_LIST:-./data/test/prompt.list}" \
  --output_dir "${TRITSP_EVAL_DIR:-./outputs/test}" \
  --recon_timesteps "${TRITSP_RECON_TIMESTEPS:-700,500,400,200,100,90,80,70,60,50,40,30,20,10}" \
  --modalities "${TRITSP_MODALITIES:-rgb,thermal}" \
  --high_boundary 300 \
  --middle_boundary 85 \
  --tritsp_cond_dim 512 \
  --tritsp_num_heads 8 \
  --tritsp_condition_tokens 12 \
  --tritsp_tsp_condition_tokens 8 \
  --tritsp_spp_condition_tokens 4 \
  --tritsp_condition_decoder_layers 2 \
  --tritsp_delta_rank 64 \
  --tritsp_gate_max 0.5 \
  --tritsp_inject_block_indices 9,14,18,23 \
  --resume \
  "${BRANCH_ARGS[@]}"
