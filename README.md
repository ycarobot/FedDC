## Disclaimer

> ⚠️ **Note: This is an ongoing research work.**

This repository is the official implementation of the paper "Towards Robust Federated Test-Time Adaptation: Dynamic Client Collaboration and Category-Aware Uncertainty", which has been submitted to *The Visual Computer*.

## Introduction
- Federated test-time adaptation (FTTA) enables privacy-preserving model adaptation to unlabeled target data during inference, yet it struggles with dynamic source client availability and uncertain test samples under distribution shifts.
  
- We propose a novel FTTA framework, termed Federated test-time adaptation under Dynamic client collaboration and Category-aware uncertainty (FedDC), which effectively improves adaptation robustness and stability. During source training, clients are aggregated adaptively based on participation history to reduce bias. At test time, category-specific thresholds separate confident and uncertain samples, preserving prediction uncertainty to mitigate noise.

## 🧠 Core Algorithm: Federated learning, test-time adaptation

This repository contains the official implementation of the FedDC framework, which is designed to address the challenge of federated test-time adaptation under dynamic client availability and distribution shifts. The core innovation lies in two collaboratively working modules:

> **Adaptive Weightingand Initialization (AWI):** Adapts client aggregation weights based on historical participation during source training, mitigating bias from unstable client availability and heterogeneous data distributions.
>
> **Selective Label Enhancement Based on Class-margin Thresholds (SLE) :** Applies category-specific uncertainty thresholds at test time to separate confident and uncertain samples, effectively suppressing noise from pseudo-labels and improving adaptation robustness.
<img width="6409" height="3359" alt="框架" src="https://github.com/user-attachments/assets/ade1e236-2f71-4c56-ba16-e6bb40f69803" />

## Requirements
- python 3.8.5
- cudatoolkit 10.2.89
- cudnn 7.6.5
- pytorch 1.11.0
- torchvision 0.12.0
- numpy 1.18.5
- tqdm 4.65.0
- matplotlib 3.7.1

If you prefer generating the CIFAR-10C, CIFAR-100C and Tiny-ImageNetC by yourself, these packages may also be required:
- wandb 0.16.0
- scikit-image 0.17.2
- opencv-python 4.8.0.74

## Install Datasets
We need users to declare a `data` to store the dataset as well as the log of training procedure. The directory structure should be :

Download the datasets used in our paper from the following links:

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Stanford Cars](https://tensorflow.google.cn/datasets/catalog/cars196)
- [PACS](https://domaingeneralization.github.io/#data)
- [CarlaTTA](https://github.com/mariodoebler/test-time-adaptation)
```
data
│       
└───dataset
│   │   CIFAR100
│       │  test
│       │  train
|       |  meta
│       │  file.txt
│   │   Tiny-ImageNet
│       │  test
│       │  train
│       │  val
│   │   PACS
│       │  art_painting
│       │  cartoon
│       │  photo
│       │  sketch
|   |   Stanford_Cars
|       |  cars_train
|       |  cars_test
|       |  cars annos.mat
|   |   Food-101
|       |  images
|       |  meta
|   |   CarlaTTA
|       |  clear  
|       |  clear_fog_1200
|       |  clear_rain_1200
|       |  clear_night_1200
|       |  clear_highway
|       |  day_night_1200
|       |  dynamic_1200
|       |  town04_dynamic_1200

```

## Run
## CIFAR-100C Experiments
We consider hybrid distribution shifts (including label shifts and feature shifts) in our CIFAR-100C experiments.

```bash
cd ./exp/cifar100/${shift}
```
- where `${shift}` should be replaced by `hybrid` (hybrid shift).

## Generate Dataset
```
./data_prepare.sh
```
This shell script will partition the CIFAR-100 dataset to 300 clients (240 source clients and 60 clients), and save the partition indices to `~/data/feddc/partition/cifar100/`. When there are corruptions (hybrid shift), we also cache the corrupted dataset to `~/data/feddc/cifar100` to save time.

## Train Global Model with FedAwi
Before running FedDC, we need to train a global model with source clients' training sets. We use FedAwi algorithm to train the global model.
```
./pretrain_fedawi_${model}.sh
```
Here `${model}` specifies the model architecture we use. We used resnet18 (ResNet-18) and vit (ViT-base/16) in our paper.

Learn Adaptation Rates
```
./feddc_train_${model}.sh
```
Learn Class-aware Margin Thresholds
```
python c_cifar100.py
```
## Fedderated Test-Time Adaptation with FedDC-batch and FedDC-online
```
./feddc_test_${model}.sh
```

You can run most of the experiments in our paper by  
shell: python main.py

Moreover, we also prepare code for various datasets and model architectures. Please check the arguments function in the `option.py` file for more details.

## Acknowledgements
This implementation is based on [ATP](https://github.com/baowenxuan/ATP) and [PASLE](https://github.com/palm-ml/PASLE).


## 引用

```
@article{FedDC,
    title={Towards Robust Federated Test-Time Adaptation: Dynamic Client Collaboration and Category-Aware Uncertainty},
    author={Yi, Changan and et al.},
    journal={submitted to The Visual Computer},
    year={2026}
}
```
