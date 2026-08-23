#!/usr/bin/env bash
# Representative paper reproduction: CIFAR-100, ResNet-18, blur -> noise.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${GPU:-0}"
data_root="${DATA_ROOT:-$repo_root/data}"
partition_root="${PARTITION_ROOT:-$data_root/atp/partition}"
seeds="${SEEDS:-0}"
methods="${METHODS:-test bn tent t3a memo em surgical tsd program}"
checkpoint="${CHECKPOINT:-$repo_root/checkpoints/cifar100_resnet18_fedavg_seed0.pkl}"
output_root="${OUTPUT_ROOT:-$repo_root/artifacts/cifar100/resnet18/blur_noise}"

partition_file="$partition_root/cifar100/client_300_partition_step_2_51_seed_0.pkl"
corruption_file="$data_root/atp/cifar100/client_300_partition_step_2_51_corruption_ood_blur_noise_random_seed_0.pkl"

for required in "$checkpoint" "$partition_file" "$corruption_file"; do
  if [ ! -f "$required" ]; then
    echo "Missing required file: $required" >&2
    echo "See DATASETS.md and exp/cifar100/hybrid/data_prepare.sh." >&2
    exit 2
  fi
done

mkdir -p "$output_root/logs" "$output_root/history"
cd "$repo_root" || exit 1

for seed in $seeds; do
  for method in $methods; do
    run_name="${method}_batch_pseed0_seed${seed}_blur_noise"
    log="$output_root/logs/${run_name}.log"
    history="$output_root/history/${run_name}.pkl"
    command=("$python_bin" main.py
      --dataset cifar100 --num_clients 300 --partition step_2_51
      --data_holdout 0.2 --client_holdout 0.2 --partition_seed 0
      --corruption ood --newco blur_noise_random --model resnet18
      --algorithm "$method" --test batch --batch_size 20 --seed "$seed"
      --data_dir "$data_root" --partition_dir "$partition_root"
      --load_model_path "$checkpoint" --history_path "$history" --save_model_path none)
    {
      echo "start_time=$(date --iso-8601=seconds)"
      printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
      CUDA_VISIBLE_DEVICES="$gpu_id" WANDB_MODE=offline "${command[@]}"
      status=$?
      echo "exit_code=$status"
      echo "end_time=$(date --iso-8601=seconds)"
    } 2>&1 | tee "$log"
  done
done

echo "Reference values: $repo_root/results/expected/cifar100_resnet18_blur_noise.csv"
