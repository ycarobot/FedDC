#!/usr/bin/env bash
# One entry point per manuscript dataset/backbone combination.
set -euo pipefail

dataset="${1:-}"
backbone="${2:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$dataset" ] || [ -z "$backbone" ]; then
  echo "Usage: bash scripts/reproduce_paper.sh <dataset> <backbone>" >&2
  exit 2
fi

case "$dataset:$backbone" in
  cifar100:resnet18)
    exec bash "$repo_root/scripts/reproduce_cifar100_r18_blur_noise.sh"
    ;;
  cifar100:resnet50|cifar100:vit|tiny_imagenet:resnet18|tiny_imagenet:resnet50|pacs_aug:resnet18|pacs_aug:resnet50|stanfordcars:resnet18|food101:resnet18)
    config="$repo_root/configs/${dataset}_${backbone}.env"
    if [ ! -f "$config" ]; then
      echo "Missing configuration: $config" >&2
      exit 2
    fi
    set -a
    # shellcheck disable=SC1090
    source "$config"
    set +a
    exec bash "$repo_root/scripts/run_three_stage.sh"
    ;;
  carlatta:deeplabv2)
    echo "CarlaTTA uses the segmentation tree documented in DATASETS.md." >&2
    echo "Set SEGMENTATION_ROOT to the test-time-adaptation segmentation directory." >&2
    test -n "${SEGMENTATION_ROOT:-}" || exit 2
    exec bash "$repo_root/scripts/reproduce_carlatta.sh"
    ;;
  *)
    echo "Unsupported combination: $dataset $backbone" >&2
    exit 2
    ;;
esac
