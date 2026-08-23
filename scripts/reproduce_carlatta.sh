#!/usr/bin/env bash
set -euo pipefail
segmentation_root="${SEGMENTATION_ROOT:?Set SEGMENTATION_ROOT}"
gpu_id="${GPU:-0}"
sequence="${SEQUENCE:-dynamic_1200.txt}"
config="${CARLATTA_CONFIG:-$segmentation_root/cfgs/feddcu.yaml}"
cd "$segmentation_root"
CUDA_VISIBLE_DEVICES="$gpu_id" python test_time.py --cfg "$config" LIST_NAME_TEST "$sequence"
