#!/usr/bin/env bash
# Generic three-stage classification runner; variables are loaded from configs/*.env.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${GPU:-0}"
data_root="${DATA_ROOT:-$repo_root/data}"
partition_root="${PARTITION_ROOT:-$data_root/atp/partition}"
seeds="${SEEDS:-0 1 2}"
output_root="${OUTPUT_ROOT:-$repo_root/artifacts/$DATASET/$BACKBONE}"
mkdir -p "$output_root/checkpoints" "$output_root/history" "$output_root/logs"
cd "$repo_root" || exit 1

run_logged() {
  local log="$1"; shift
  {
    echo "start_time=$(date --iso-8601=seconds)"
    printf 'command='; printf '%q ' "$@"; printf '\n'
    CUDA_VISIBLE_DEVICES="$gpu_id" WANDB_MODE=offline "$@"
    echo "exit_code=$?"
    echo "end_time=$(date --iso-8601=seconds)"
  } 2>&1 | tee "$log"
}

for seed in $seeds; do
  model_path="$output_root/checkpoints/fedawi_${BACKBONE}_pseed${PARTITION_SEED}_seed${seed}.pkl"
  adapt_path="$output_root/history/adapt_${BACKBONE}_pseed${PARTITION_SEED}_seed${seed}.pkl"
  common=(--dataset "$DATASET" --num_clients "$NUM_CLIENTS" --partition "$PARTITION"
    --data_holdout 0.2 --client_holdout "$CLIENT_HOLDOUT" --partition_seed "$PARTITION_SEED"
    --corruption "$CORRUPTION" --newco "$NEWCO" --model "$BACKBONE" --batch_size 20
    --seed "$seed" --data_dir "$data_root" --partition_dir "$partition_root")

  run_logged "$output_root/logs/pretrain_seed${seed}.log" "$python_bin" main.py "${common[@]}" \
    --algorithm fedawi --gm_rounds 200 --part_rate 1.0 --lm_lr 0.01 --lm_epochs 1 \
    --fluctuate_type 5 --pre_train_init "$PRETRAIN_INIT" \
    --history_path none --save_model_path "$model_path"

  run_logged "$output_root/logs/adapt_seed${seed}.log" "$python_bin" main.py "${common[@]}" \
    --algorithm atp --gm_rounds 200 --part_rate 0.25 --lm_lr 0.01 --lm_epochs 1 \
    --load_model_path "$model_path" --history_path "$adapt_path" --save_model_path none

  for mode in batch online; do
    run_logged "$output_root/logs/feddcu_${mode}_seed${seed}.log" "$python_bin" main.py "${common[@]}" \
      --algorithm feddc --test "$mode" --load_model_path "$model_path" \
      --load_adapt_path "$adapt_path" --class_threshold_path "${CLASS_THRESHOLD_PATH:-none}" \
      --history_path "$output_root/history/feddcu_${mode}_seed${seed}.pkl" --save_model_path none
  done
done
