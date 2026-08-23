# Checkpoint manifest

| Field | Value |
|---|---|
| Published name | `cifar100_resnet18_fedavg_seed0.pkl` |
| Original experiment | CIFAR-100, ResNet-18, FedAvg, hybrid shift |
| Partition | 300 clients, `step_2_51`, partition seed 0 |
| Training seed | 0 |
| Container | PyTorch-compatible pickled `OrderedDict` |
| State entries | 122 |
| Size | 44,988,357 bytes |
| SHA-256 | `89f7fb7973e69573076a03c4e8ee467db67a75a34ccaa8d124e89121c7546410` |

The public filename is intentionally concise. The original local filename was `pretrain_fedavg_resnet18_pseed_0_seed_0.pkl`; the original file was not moved or modified. The published copy stores the same tensors on CPU so it can be loaded on both CPU-only and CUDA systems.

Verify before use:

```bash
sha256sum checkpoints/cifar100_resnet18_fedavg_seed0.pkl
```

This checkpoint is provided for Source and baseline sanity checks. Exact FedDCU results also require the learned adaptation-rate history and category thresholds, which are produced by stages 2 and 3 of the reproduction scripts.
