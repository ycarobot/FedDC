# Reproducibility

## Frozen release

Use release tag `v0.1.0-repro`. Its exact immutable commit is `960656ab0fa2ba89720bdc39e463a09450040482`. Verify it with:

```bash
git rev-list -n 1 v0.1.0-repro
git status --short
```

The second command should print nothing. The archived software release is identified by Zenodo DOI [`10.5281/zenodo.22072543`](https://doi.org/10.5281/zenodo.22072543).

## Determinism and random seeds

All reported classification experiments use seeds `0`, `1`, and `2` for partition generation and client-unavailability simulation. The representative public command defaults to seed `0`; set `SEEDS="0 1 2"` to execute all three.

`main.py` seeds Python, NumPy, CPU/CUDA PyTorch generators and enables deterministic cuDNN kernels. Exact bitwise equality can still depend on GPU architecture, CUDA/cuDNN version, and nondeterministic third-party kernels.

## Three-stage protocol

1. Pre-train a global model for 200 communication rounds with local learning rate 0.01, one local epoch, batch size 20, participation pattern 5, and the dataset-specific initialization factor.
2. Learn layer-wise adaptation rates for 200 rounds with local learning rate 0.01 and batch size 20.
3. Evaluate batch and online adaptation with batch size 20, buffer capacity 5, threshold decrement 0.1, and desired threshold floor 0.001.

CarlaTTA uses DeepLabV2/ResNet-101, 100 pre-training rounds, source batch size 2, and online batch size 1. See `DATASETS.md` for split details.

## Hardware and runtime

Reference hardware: one NVIDIA GeForce RTX 3090 Ti (24 GiB), Ubuntu Linux, CUDA 10.2 and cuDNN 7.6.5. The scripts accept `GPU=<index>` and run one experiment per GPU serially.

Approximate wall-clock times depend on storage and corruption caching:

| Stage | CIFAR-100/ResNet-18 | Tiny-ImageNet/ResNet-18 | PACS/ResNet-18 | CarlaTTA/DeepLabV2 |
|---|---:|---:|---:|---:|
| Data preparation | 10–30 min | 30–90 min | 10–30 min | supplied split files |
| Global pre-training | 3–6 h | 6–12 h | 1–3 h | 8–16 h |
| Adaptation-rate training | 2–5 h | 5–10 h | 1–3 h | 6–12 h |
| One test setting | 2–10 min | 5–20 min | 2–10 min | 10–30 min |

These are planning ranges, not performance claims. Each script writes its command, start/end time, seed, checkpoint paths, and exit code to `artifacts/logs/`.

## Representative expected output

```bash
DATA_ROOT=/path/to/data GPU=0 SEEDS="0 1 2" \
  bash scripts/reproduce_cifar100_r18_blur_noise.sh
```

Reference manuscript values are stored in `results/expected/cifar100_resnet18_blur_noise.csv` and the adjacent `.log`. Small deviations are expected across hardware/software stacks. The table values are reference outputs, not runtime assertions.

## Manuscript-to-code map

| Manuscript item | Dataset/backbone | Entry point | Expected artifact |
|---|---|---|---|
| Table 1 / `tab_1` | CIFAR-100, ResNet-18/50, ViT-B/16 | `scripts/reproduce_paper.sh cifar100 <backbone>` | `artifacts/cifar100/<backbone>/` |
| Table 2 / `tab_2` | Tiny-ImageNet, ResNet-18/50 | `scripts/reproduce_paper.sh tiny_imagenet <backbone>` | `artifacts/tiny_imagenet/<backbone>/` |
| Table 3 / `tab_3` | Food-101 and Stanford Cars, ResNet-18 | `scripts/reproduce_paper.sh <food101|stanfordcars> resnet18` | corresponding artifact directory |
| Table 4 / `tab_4` | PACS, ResNet-18/50 | `scripts/reproduce_paper.sh pacs_aug <backbone>` | `artifacts/pacs_aug/<backbone>/` |
| Table 5 / `tab_5` and qualitative figure | CarlaTTA, DeepLabV2 | `scripts/reproduce_paper.sh carlatta deeplabv2` | `artifacts/carlatta/deeplabv2/` |
| Table 6 / `tab_6` | client-unavailability ablation | set `FLUCTUATE_TYPES="3 4 5 6"` | per-pattern logs |
| module ablation / `tab:module_ablation_buffer` | CIFAR-100, ResNet-18 | `ABLATION=modules scripts/reproduce_paper.sh cifar100 resnet18` | `artifacts/ablations/` |

The compact release includes the complete command mapping but only one checkpoint and one expected-output group. This avoids redistributing raw datasets and hundreds of megabytes of redundant weights.

## Output conventions

- Checkpoints: `artifacts/<dataset>/<backbone>/checkpoints/`
- Adaptation rates: `artifacts/<dataset>/<backbone>/history/`
- Logs: `artifacts/<dataset>/<backbone>/logs/`
- Tables: `artifacts/<dataset>/<backbone>/tables/`
- File names contain dataset, backbone, partition seed, test seed, corruption pair, and adaptation mode.
