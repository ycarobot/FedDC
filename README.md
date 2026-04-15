# Towards Robust Federated Test-Time Adaptation: Dynamic Client Collaboration and Category-Aware Uncertainty
These codes are directly related to the current manuscript submitted to The Visual Computer: Towards Robust Federated Test-Time Adaptation: Dynamic Client Collaboration and Category-Aware Uncertainty.

## Introduction
- Federated test-time adaptation (FTTA) enables privacy-preserving model adaptation to unlabeled target data during inference, yet it struggles with dynamic source client availability and uncertain test samples under distribution shifts.
  
- We propose a novel FTTA framework, termed Federated test-time adaptation under Dynamic client collaboration and Category-aware uncertainty (FedDC), which effectively improves adaptation robustness and stability. During source training, clients are aggregated adaptively based on participation history to reduce bias. At test time, category-specific thresholds separate confident and uncertain samples, preserving prediction uncertainty to mitigate noise.
  

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
We consider hybrid distribution shifts (including label shifts and feature shifts) in our CIFAR-10C experiments.

```bash
cd ./exp/cifar100/${shift}
```
- where `${shift}` should be replaced by `hybrid` (hybrid shift).

## Generate Dataset
```
./data_prepare.sh
```


## Experiment

You can run most of the experiments in our paper by  
shell: python main.py

Moreover, we also prepare code for various datasets and model architectures. Please check the arguments function in the `main.py` file for more details.

## Acknowledgements
This implementation is based on [ATP](https://github.com/baowenxuan/ATP) and [PASLE](https://github.com/palm-ml/PASLE).

