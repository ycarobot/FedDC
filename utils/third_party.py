import numpy as np
import torch
import torch.nn as nn

from PIL import ImageOps, Image
from torchvision import transforms


## https://github.com/google-research/augmix

# def _augmix_aug(x_orig):
#     x_orig = preaugment(x_orig)
#     x_processed = preprocess(x_orig)
#     w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
#     m = np.float32(np.random.beta(1.0, 1.0))
#
#     mix = torch.zeros_like(x_processed)
#     for i in range(3):
#         x_aug = x_orig.copy()
#         for _ in range(np.random.randint(1, 4)):
#             x_aug = np.random.choice(augmentations)(x_aug)
#         mix += w[i] * preprocess(x_aug)
#     mix = m * x_processed + (1 - m) * mix
#     return mix


def _vanilla_aug_cifar10(x_orig):
    x_orig = preaugment_cifar(x_orig)
    x_processed = preprocess(x_orig)
    return x_processed


def _vanilla_aug_digit(x_orig):
    x_orig = preaugment_digit(x_orig)
    x_processed = preprocess(x_orig)
    return x_processed


def _vanilla_aug_pacs(x_orig):
    x_orig = preaugment_pacs(x_orig)
    x_processed = preprocess(x_orig)
    return x_processed


# 新增：Tiny-ImageNet的基础增强函数
def _vanilla_aug_tiny_imagenet(x_orig):
    x_orig = preaugment_tiny_imagenet(x_orig)  # 使用Tiny-ImageNet专属的预处理增强
    x_processed = preprocess(x_orig)  # 统一转换为Tensor
    return x_processed


# 新增：Food101的基础增强函数（适配224x224标准尺寸）
def _vanilla_aug_food101(x_orig):
    x_orig = preaugment_food101(x_orig)  # Food101专属预处理增强
    x_processed = preprocess(x_orig)     # 统一转换为Tensor
    return x_processed


# 新增：StanfordCars的基础增强函数（适配224x224标准尺寸）
def _vanilla_aug_stanfordcars(x_orig):
    x_orig = preaugment_stanfordcars(x_orig)  # StanfordCars专属预处理增强
    x_processed = preprocess(x_orig)          # 统一转换为Tensor
    return x_processed


# aug = _augmix_aug
aug_cifar = _vanilla_aug_cifar10
aug_digit = _vanilla_aug_digit
aug_pacs = _vanilla_aug_pacs
aug_tiny_imagenet = _vanilla_aug_tiny_imagenet  # 注册Tiny-ImageNet增强函数
aug_food101 = _vanilla_aug_food101              # 注册Food101增强函数
aug_stanfordcars = _vanilla_aug_stanfordcars    # 注册StanfordCars增强函数


def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)


def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)


def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    # 适配不同尺寸：优先按输入图像尺寸动态调整，兼容所有数据集
    img_size = pil_img.size[0]  # 获取图像宽度（假设正方形）
    return pil_img.transform((img_size, img_size), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)


def shear_y(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    # 动态适配图像尺寸
    img_size = pil_img.size[0]
    return pil_img.transform((img_size, img_size), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR, fillcolor=128)


def translate_x(pil_img, level):
    # 动态计算平移幅度（按图像尺寸的1/3）
    img_size = pil_img.size[0]
    level = int_parameter(rand_lvl(level), img_size / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((img_size, img_size), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)


def translate_y(pil_img, level):
    # 动态计算平移幅度
    img_size = pil_img.size[0]
    level = int_parameter(rand_lvl(level), img_size / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((img_size, img_size), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=128)


def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


augmentations = [
    autocontrast,
    equalize,
    lambda x: rotate(x, 1),
    lambda x: solarize(x, 1),
    lambda x: shear_x(x, 1),
    lambda x: shear_y(x, 1),
    lambda x: translate_x(x, 1),
    lambda x: translate_y(x, 1),
    lambda x: posterize(x, 1),
]

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

preaugment_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
    transforms.RandomHorizontalFlip(),
])

preaugment_digit = transforms.Compose([
    transforms.RandomCrop(28, padding=4, padding_mode="symmetric"),
    transforms.RandomHorizontalFlip(),
])

preaugment_pacs = transforms.Compose([
    transforms.RandomCrop(224, padding=32, padding_mode="symmetric"),
    transforms.RandomHorizontalFlip(),
])

# 新增：Tiny-ImageNet的预处理增强（适配64x64尺寸）
preaugment_tiny_imagenet = transforms.Compose([
    # Tiny-ImageNet图像尺寸为64x64，设置合适的裁剪和填充
    transforms.RandomCrop(64, padding=8, padding_mode="symmetric"),  # 填充8像素后随机裁剪64x64
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
])

# 新增：Food101的预处理增强（适配224x224标准尺寸，与PACS保持一致）
# Food101数据集图像通常缩放为224x224，增强策略适配分类任务
preaugment_food101 = transforms.Compose([
    transforms.RandomCrop(224, padding=32, padding_mode="symmetric"),  # 填充32像素后随机裁剪224x224
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（食物图像水平翻转不影响分类）
    transforms.RandomRotation(15),  # 小幅随机旋转（适配食物摆放角度）
])

# 新增：StanfordCars的预处理增强（适配224x224标准尺寸）
# 车辆图像增强：减少旋转幅度（车辆方向重要），保留水平翻转
preaugment_stanfordcars = transforms.Compose([
    transforms.RandomCrop(224, padding=32, padding_mode="symmetric"),  # 填充32像素后随机裁剪224x224
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（车辆左右翻转不影响分类）
    transforms.RandomRotation(5),  # 小幅旋转（车辆旋转过大影响分类，仅5度）
])