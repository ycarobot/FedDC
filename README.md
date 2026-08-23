# FedDCU: Federated Test-Time Adaptation under Dynamic Client Collaboration and Category-Aware Uncertainty

Official implementation of **FedDCU**, a federated test-time adaptation framework for dynamic source-client participation and category-aware uncertainty.

## Reproduce paper results

The reproducibility entry point is [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). It records the release, seeds, hardware, expected runtime, checkpoints, expected outputs, and the mapping from manuscript tables/figures to scripts.

```bash
git clone --branch v0.1.0-repro https://github.com/ycarobot/FedDCU.git
cd FedDCU
conda env create -f environment.yml
conda activate feddcu
```

Representative CIFAR-100/ResNet-18 hybrid-shift experiment:

```bash
DATA_ROOT=/path/to/data GPU=0 bash scripts/reproduce_cifar100_r18_blur_noise.sh
```

One-command dataset/backbone entry points:

```bash
bash scripts/reproduce_paper.sh cifar100 resnet18
bash scripts/reproduce_paper.sh cifar100 resnet50
bash scripts/reproduce_paper.sh cifar100 vit
bash scripts/reproduce_paper.sh tiny_imagenet resnet18
bash scripts/reproduce_paper.sh tiny_imagenet resnet50
bash scripts/reproduce_paper.sh pacs_aug resnet18
bash scripts/reproduce_paper.sh pacs_aug resnet50
bash scripts/reproduce_paper.sh stanfordcars resnet18
bash scripts/reproduce_paper.sh food101 resnet18
bash scripts/reproduce_paper.sh carlatta deeplabv2
```

The compact public bundle contains one representative FedAvg checkpoint and one expected-output example. The remaining commands reproduce results from training rather than requiring every large artifact to be stored in Git.

## Method

FedDCU contains three stages:

1. **Source pre-training:** participation-aware adaptive weighting and initialization reduce bias caused by dynamic client availability.
2. **Adaptation-rate training:** layer-wise adaptation rates are learned using unlabeled target-client data.
3. **Test-time adaptation:** category-aware margin thresholds separate confident predictions from uncertain candidate labels and reduce pseudo-label error accumulation.

## Datasets

Official and fallback links, licenses, expected directory names, preprocessing, and partition settings are listed in [`DATASETS.md`](DATASETS.md). Raw datasets and generated corruptions are not redistributed.

| Dataset | Official | Fallback |
|---|---|---|
| CIFAR-100 | [Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) | [Hugging Face](https://huggingface.co/datasets/uoft-cs/cifar100) |
| Tiny-ImageNet | [Stanford](http://cs231n.stanford.edu/tiny-imagenet-200.zip) | [Hugging Face](https://huggingface.co/datasets/galilai-group/tiny-imagenet) |
| Food-101 | [ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) | [Hugging Face](https://huggingface.co/datasets/ethz/food101) |
| Stanford Cars | [Cars196/TFDS](https://www.tensorflow.org/datasets/catalog/cars196) | [Hugging Face](https://huggingface.co/datasets/tanganke/stanford_cars) |
| PACS | [DG benchmark](https://domaingeneralization.github.io/#data) | [mirror instructions](DATASETS.md#pacs) |
| CarlaTTA | [Benchmark repository](https://github.com/mariodoebler/test-time-adaptation#segmentation) | [Google Drive archives linked by the benchmark](https://github.com/mariodoebler/test-time-adaptation#carlatta) |

## Released artifact

The representative checkpoint is [`checkpoints/cifar100_resnet18_fedavg_seed0.pkl`](checkpoints/cifar100_resnet18_fedavg_seed0.pkl). Its provenance, checksum, and compatible command are recorded in [`checkpoints/MANIFEST.md`](checkpoints/MANIFEST.md).

## Environment

The paper environment used Python 3.8.5, PyTorch 1.11.0, torchvision 0.12.0, CUDA 10.2, and cuDNN 7.6.5. See [`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt). Hardware and runtime notes are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md#hardware-and-runtime).

## Version and citation

- Reproducibility release: `v0.1.0-repro`
- Exact release commit: resolve with `git rev-list -n 1 v0.1.0-repro`
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Zenodo DOI: pending repository-to-Zenodo archival; no DOI is claimed before Zenodo issues one.

```bibtex
@article{FedDCU2026,
  title   = {FedDCU: Federated Test-Time Adaptation under Dynamic Client Collaboration and Category-Aware Uncertainty},
  author  = {Li, Yongcai and Zhou, Yuexia and Liu, Xiangyu and Chen, Kai and Chen, Jinpeng and Yi, Chang'an},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Code release v0.1.0-repro}
}
```

## Acknowledgements

This implementation builds on [ATP](https://github.com/baowenxuan/ATP), [PASLE](https://github.com/palm-ml/PASLE), and the [test-time-adaptation benchmark](https://github.com/mariodoebler/test-time-adaptation).

## License

Released under the [MIT License](LICENSE). Dataset licenses remain with their respective owners.
