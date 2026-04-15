import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Dict, List, Tuple

# 从代码3导入损坏函数
from .distortions import distortions, test_distortions


def make_tinyimagenet_c(partition_idxs: Dict[int, List[int]],
                        is_train: Dict[int, bool],
                        data_dir: str = '../data',
                        mode: str = "none") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    生成TinyImageNet-C数据集（最终修复版）

    参数:
        partition_idxs: 字典{客户端ID: 样本索引列表}
        is_train: 字典{客户端ID: 是否为训练客户端}
        data_dir: 数据集根目录
        mode: 'iid'或'ood'
            'iid' - 源客户端和目标客户端使用相同的15种损坏
            'ood' - 源客户端用15种损坏，目标客户端用4种额外损坏

    返回:
        (图像张量, 标签张量, 损坏信息字典)
    """
    # 1. 初始化路径和参数
    root = os.path.join('/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/data', 'tiny-imagenet-200')
    TARGET_SIZE = (64, 64)  # TinyImageNet标准尺寸

    # 2. 创建类别ID映射字典
    id_dict = {}
    with open(os.path.join(root, 'wnids.txt'), 'r') as f:
        for i, line in enumerate(f):
            id_dict[line.strip()] = i

    # 3. 定义转换流程
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),  # 强制调整尺寸
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # 确保3通道
    ])

    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])

    # 4. 加载数据集
    def load_dataset_files(root_dir):
        files = []
        labels = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name, "images")
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.JPEG'):
                    files.append(os.path.join(class_dir, img_name))
                    labels.append(id_dict[class_name])
        return files, labels

    # 加载训练集
    train_files, train_labels = load_dataset_files(os.path.join(root, "train"))

    # 加载验证集
    val_files = []
    val_labels = []
    val_img_to_class = {}
    with open(os.path.join(root, "val", "val_annotations.txt"), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name, class_name = parts[0], parts[1]
                val_img_to_class[img_name] = class_name

    val_img_dir = os.path.join(root, "val", "images")
    for img_name in os.listdir(val_img_dir):
        if img_name.endswith('.JPEG') and img_name in val_img_to_class:
            val_files.append(os.path.join(val_img_dir, img_name))
            val_labels.append(id_dict[val_img_to_class[img_name]])

    # 合并训练集和验证集
    all_files = train_files + val_files
    all_labels = train_labels + val_labels
    num_samples = len(all_files)

    # 5. 初始化容器
    corrupted_images = [None] * num_samples
    final_labels = [None] * num_samples
    corruption_info = {}

    # 6. 处理每个客户端
    for cid, sids in tqdm(partition_idxs.items()):
        # 选择损坏类型
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)

        corruption_info[cid] = (distortion.__name__, severity)

        # 对客户端样本应用损坏
        for sid in sids:
            if sid >= num_samples:
                continue

            # 加载图像
            img_path = all_files[sid]
            try:
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # 应用预处理转换
                image_tensor = transform(image)
                image_pil = transforms.ToPILImage()(image_tensor)

                # 应用损坏（特殊处理fog、frost和snow）
                try:
                    if distortion.__name__ in ['fog', 'frost', 'snow']:
                        corrupted_img = apply_special_distortion(
                            distortion.__name__,
                            image_pil,
                            severity,
                            target_size=TARGET_SIZE
                        )
                    else:
                        corrupted_img = distortion(image_pil, severity=severity)

                    # 转换为numpy数组
                    if isinstance(corrupted_img, Image.Image):
                        corrupted_img = np.array(corrupted_img)
                    elif isinstance(corrupted_img, torch.Tensor):
                        corrupted_img = corrupted_img.numpy()

                    # 确保3通道
                    if corrupted_img.ndim == 2:
                        corrupted_img = np.stack([corrupted_img] * 3, axis=-1)
                    elif corrupted_img.shape[-1] == 1:
                        corrupted_img = np.repeat(corrupted_img, 3, axis=-1)

                    # 确保尺寸正确
                    if corrupted_img.shape[0] != TARGET_SIZE[0] or corrupted_img.shape[1] != TARGET_SIZE[1]:
                        corrupted_img = np.array(Image.fromarray(corrupted_img).resize(TARGET_SIZE))

                    # 转换为uint8
                    corrupted_img = np.uint8(corrupted_img)

                    # 应用后处理转换
                    corrupted_img = Image.fromarray(corrupted_img)
                    corrupted_img = post_transform(corrupted_img)

                except Exception as e:
                    print(f"应用损坏 {distortion.__name__} 时出错: {e}")
                    # 损坏失败时使用原始图像
                    corrupted_img = post_transform(image_pil)

            except Exception as e:
                print(f"加载图像 {img_path} 时出错: {e}")
                # 图像加载失败时使用黑色图像
                corrupted_img = torch.zeros(3, *TARGET_SIZE)

            # 最终尺寸检查
            if corrupted_img.shape[1:] != torch.Size(TARGET_SIZE):
                corrupted_img = transforms.functional.resize(corrupted_img, TARGET_SIZE)

            corrupted_images[sid] = corrupted_img
            final_labels[sid] = all_labels[sid]

    # 7. 最终验证和返回
    corrupted_images = torch.stack(corrupted_images)
    final_labels = torch.LongTensor(final_labels)

    return corrupted_images, final_labels, corruption_info


def apply_special_distortion(distortion_name, image, severity, target_size=(64, 64)):
    """
    特殊处理fog、frost和snow损坏
    """
    # 转换为numpy数组并归一化到[0,1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    if distortion_name == 'fog':
        # 自定义fog实现
        c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]

        # 生成fog模式
        fog_pattern = generate_fog_pattern(target_size, decay=c[1])

        # 应用fog效果
        img_array = img_array + c[0] * fog_pattern
        img_array = img_array * img_array.max() / (img_array.max() + c[0])

    elif distortion_name == 'frost':
        # 自定义frost实现
        frost_pattern = generate_frost_pattern(target_size, severity)
        img_array = np.clip(img_array + frost_pattern * 0.8, 0, 1)

    elif distortion_name == 'snow':
        # 自定义snow实现
        snow_layer = generate_snow_layer(target_size, severity)
        img_array = np.clip(img_array + snow_layer, 0, 1)

    # 转换回PIL图像
    return Image.fromarray((img_array * 255).astype(np.uint8))


def generate_fog_pattern(size, decay=3):
    """
    生成fog模式
    """
    # 使用分形噪声生成fog模式
    x = np.arange(size[0])
    y = np.arange(size[1])
    xx, yy = np.meshgrid(x, y)

    # 创建基础噪声
    noise = np.random.rand(size[0], size[1])

    # 应用分形衰减
    for level in range(1, 5):
        scale = 2 ** level
        noise += np.random.rand(size[0] // scale, size[1] // scale).repeat(scale, axis=0).repeat(scale, axis=1) / (
                    decay ** level)

    # 归一化到[0,1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise[..., np.newaxis]  # 添加通道维度


def generate_frost_pattern(size, severity):
    """
    生成frost模式
    """
    # 根据严重程度选择frost类型
    frost_types = [
        lambda: np.random.rand(size[0], size[1], 1) * 0.5,  # 轻度霜冻
        lambda: np.random.rand(size[0], size[1], 1) * 0.7,  # 中度霜冻
        lambda: np.random.rand(size[0], size[1], 1) * 0.9,  # 重度霜冻
        lambda: np.ones((size[0], size[1], 1)) * 0.6,  # 均匀霜冻
        lambda: np.random.rand(size[0], size[1], 1) > 0.3  # 斑驳霜冻
    ]

    return frost_types[severity - 1]()


def generate_snow_layer(size, severity):
    """
    生成snow层
    """
    # 创建基础snow层
    snow_layer = np.zeros((size[0], size[1], 1))

    # 根据严重程度调整snow密度
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    density = densities[severity - 1]

    # 生成随机snow点
    mask = np.random.rand(size[0], size[1]) < density
    snow_layer[mask] = np.random.rand(mask.sum(), 1) * 0.8 + 0.2

    return snow_layer


def random_distortion(range_severity=(1, 6), mode='train'):
    """
    随机选择损坏类型和严重程度
    """
    if mode == "train" or mode == "iid":
        selected_distortions = distortions  # 15种标准损坏
        # s=1
    elif mode == "ood":
        selected_distortions = test_distortions  # 4种额外损坏
        # s=5
    else:
        raise ValueError(f"未知模式: {mode}")

    # 随机选择损坏和严重程度
    num_distortions = len(selected_distortions)
    i = np.random.randint(num_distortions)
    s = np.random.randint(low=range_severity[0], high=range_severity[1])

    return selected_distortions[i], s


def test():
    """测试函数"""
    partition_idxs = {
        0: list(range(10)),  # 客户端0的样本索引
        1: list(range(10, 20))  # 客户端1的样本索引
    }
    is_train = {
        0: True,  # 源客户端
        1: False  # 目标客户端
    }

    # 生成损坏后的数据集
    images, labels, corruption_info = make_tinyimagenet_c(
        partition_idxs=partition_idxs,
        is_train=is_train,
        data_dir='../data',
        mode="ood"
    )

    print("测试成功！")
    print(f"图像张量形状: {images.shape}")
    print(f"标签张量形状: {labels.shape}")
    print(f"损坏信息: {corruption_info}")


# if __name__ == '__main__':
#     test()