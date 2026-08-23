# Dataset acquisition and layout

Raw datasets are not distributed in this repository. Verify each dataset's license before use. `DATA_ROOT` below is an arbitrary local directory.

| Dataset | Official source | Fallback source | Expected directory |
|---|---|---|---|
| CIFAR-100 | https://www.cs.toronto.edu/~kriz/cifar.html | https://huggingface.co/datasets/uoft-cs/cifar100 | `$DATA_ROOT/torchvision/cifar-100-python` |
| Tiny-ImageNet-200 | http://cs231n.stanford.edu/tiny-imagenet-200.zip | https://huggingface.co/datasets/galilai-group/tiny-imagenet | `$DATA_ROOT/Tiny-ImageNet` |
| Food-101 | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ | https://huggingface.co/datasets/ethz/food101 | `$DATA_ROOT/torchvision/food-101` |
| Stanford Cars | https://www.tensorflow.org/datasets/catalog/cars196 | https://huggingface.co/datasets/tanganke/stanford_cars | `$DATA_ROOT/Stanford_Cars` |
| PACS | https://domaingeneralization.github.io/#data | PACS download referenced by DomainBed/DG benchmark | `$DATA_ROOT/PACS` |
| CarlaTTA | https://github.com/mariodoebler/test-time-adaptation#segmentation | individual Google Drive archives under the benchmark's CarlaTTA section | `$DATA_ROOT/CarlaTTA` |

Fallback mirrors may package files differently. Convert them to the expected original directory layout; do not mix train/test flags or transformed images across providers.

## CIFAR-100

- 300 clients: 240 source and 60 target clients.
- Source client: 160 train and 40 validation samples; target client: 200 unlabeled test samples.
- Step partition: two major classes with 51 images each and 98 minor classes with one image each.
- Corruption categories: Noise, Blur, Weather, Digital; severity sampled from 1–5.
- Seeds: `0,1,2`; partition files are fixed within each partition seed.

## Tiny-ImageNet

- 200 clients: 160 source and 40 target clients; source holdout ratio 0.2.
- Step partition: two major classes with 176 images each and 198 minor classes with one image each.
- Resize/crop to 64×64 and normalize with mean `(0.4802,0.4481,0.3975)` and standard deviation `(0.2302,0.2265,0.2262)`.
- Seeds: `0,1,2`.

## PACS

- Leave-one-domain-out across Art Painting, Cartoon, Photo and Sketch.
- Seven clients per domain: 21 source and seven target clients in each run.
- Two majority and five minority classes with an approximate 16:1 ratio; source holdout ratio 0.2.
- Resize to 224×224 and use ImageNet normalization.
- Seeds: `0,1,2`.

## Food-101 and Stanford Cars

- Food-101: 200 clients (160 source/40 target), step partition `step_2_203`.
- Stanford Cars: 28 clients (22 source/six target), step partition `step_2_192` using the original `cars_annos.mat` train/test flags.
- Resize to 256, crop to 224 and use ImageNet normalization.
- Seeds: `0,1,2`.

## CarlaTTA

- DeepLabV2 with a ResNet-101 backbone and 14 evaluation classes; ignore label 255.
- Eleven clear-weather source clients identified by Town10HD IDs `10,11,12,13,14,15,20,21,22,23,24`.
- Test sequences: day2night, clear2fog, clear2rain, dynamic, and highway.
- Follow the benchmark names `clear_train.txt`, `day_night_1200.txt`, `clear_fog_1200.txt`, `clear_rain_1200.txt`, `dynamic_1200.txt`, and `town04_dynamic_1200.txt`.
