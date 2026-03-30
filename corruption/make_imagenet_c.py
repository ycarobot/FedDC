
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os

from .distortions import distortions, test_distortions


# def make_imagenet_c(partition_idxs, is_train, data_dir='/mnt/sda/PythonProject/Learning_Dataset', mode="none"):
#     """
#     Generate TensorDataset of ImageNet-C
#         partition_idxs: dictionary of (client index : list of sample indices)
#         is_train: whether each client is source client or target client
#         data_dir: root of torchvision dataset
#         mode: 'iid' or 'ood'.
#             'iid' uses the same 15 distortions for both source and target clients
#             'ood' uses 15 distortions for source client and 4 additional distortions for target clients
#     """
#     # 只使用验证集
#     val_dir = os.path.join(data_dir, 'ImageNet_val/val')
#
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.ToPILImage(),  # no normalization
#     ])
#
#     post_transform = transforms.Compose([
#         transforms.ToPILImage(),  # this is important
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     # 只加载验证集
#     dataset = datasets.ImageFolder(root=val_dir, transform=transform)
#
#     imagenet_c, labels = [None] * len(dataset), [None] * len(dataset)
#
#     corruption = {}
#
#     for cid, sids in tqdm(partition_idxs.items()):
#         if is_train[cid]:
#             distortion, severity = random_distortion(mode="train")
#         else:
#             distortion, severity = random_distortion(mode=mode)
#         corruption[cid] = (distortion.__name__, severity)
#         for sid in sids:
#             x, y = dataset[sid]
#             x = distortion(x, severity=severity)  # add distortion
#             x = np.uint8(x)  # convert back to original space
#             x = post_transform(x)
#             imagenet_c[sid] = x
#             labels[sid] = y
#     assert None not in imagenet_c
#     assert None not in labels
#
#     imagenet_c = torch.stack(imagenet_c)
#     labels = torch.LongTensor(labels)
#
#     return imagenet_c, labels, corruption

class ImageNetCDataset(Dataset):
    """封装ImageNet-C数据集的自定义Dataset类"""
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor
        assert len(self.data) == len(self.labels), "数据和标签长度不匹配"

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def make_imagenet_c(partition_idxs, is_train, data_dir='/mnt/sda/PythonProject/Learning_Dataset', mode="none"):
    """
    加载并处理已有的ImageNet-C数据集（非动态生成）
    Args:
        partition_idxs: 字典，格式为 {客户端ID: 样本索引列表}，表示每个客户端分配的样本
        is_train: 列表/字典，指示每个客户端是源客户端（True）还是目标客户端（False）
        data_dir: 数据集根目录，ImageNet-C应存储在此目录下
        mode: 'iid' 或 'ood'，用于区分客户端使用的失真类型集合
    Returns:
        data_tensor: 拼接后的图像张量 (总样本数, 3, 224, 224)
        labels_tensor: 对应的标签张量 (总样本数,)
        corruption: 字典，记录每个客户端使用的失真类型和强度
    """
    # ImageNet-C的标准目录结构：{data_dir}/imagenet-c/{corruption}/{severity}/...
    # 先定义所有可能的失真类型（与ImageNet-C保持一致）
    all_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    ood_corruptions = ['speckle_noise','gaussian_blur','saturate','spatter']  # 可根据需求添加额外的OOD失真类型

    # 定义图像预处理（与ImageNet标准预处理一致）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])   # ImageNet标准差
    ])

    # 根据mode选择可用的失真类型
    if mode == "train" or mode == "iid":
        selected_corruptions = all_corruptions
    elif mode == "ood":
        selected_corruptions = ood_corruptions
    else:
        raise NotImplementedError(f"不支持的模式: {mode}")

    # 收集所有客户端需要用到的样本
    # 先获取数据集总长度（通过遍历所有失真和强度的样本数）
    # 注意：实际使用时可预先计算总样本数，或通过索引映射避免内存占用过高
    total_samples = sum(len(sids) for sids in partition_idxs.values())
    data_list = [None] * total_samples  # 用列表暂存每个样本，避免提前分配大张量
    labels_list = [None] * total_samples
    corruption = {}  # 记录每个客户端的失真信息

    # 遍历每个客户端，分配对应的失真类型、强度和样本
    for cid, sids in tqdm(partition_idxs.items(), desc="加载ImageNet-C数据"):
        # 为当前客户端随机选择失真类型和强度（1-5级）
        if is_train[cid]:
            # 源客户端：从训练集失真中选择
            corruption_type = np.random.choice(all_corruptions)
        else:
            # 目标客户端：根据mode选择
            corruption_type = np.random.choice(selected_corruptions)
        severity = np.random.randint(1, 6)  # ImageNet-C强度为1-5级
        corruption[cid] = (corruption_type, severity)

        # 构建当前失真和强度的数据集路径
        corrupt_dir = os.path.join(data_dir, 'ImageNet-C', corruption_type, str(severity))
        if not os.path.exists(corrupt_dir):
            raise FileNotFoundError(f"ImageNet-C路径不存在: {corrupt_dir}")

        # 加载该失真类型下的图像（使用ImageFolder，假设标签与ImageNet一致）
        # 注意：ImageNet-C的标签与原始ImageNet验证集一致，因此可通过文件夹结构读取
        corrupt_dataset = datasets.ImageFolder(
            root=corrupt_dir,
            transform=transform
        )

        # 为当前客户端的每个样本分配数据
        for idx_in_client, sid in enumerate(sids):
            # 从当前失真数据集中随机选择样本（确保索引不越界）
            sample_idx = np.random.randint(len(corrupt_dataset))
            img, label = corrupt_dataset[sample_idx]
            data_list[sid] = img
            labels_list[sid] = label

    # 验证所有样本均已正确加载
    assert None not in data_list, "存在未加载的样本"
    assert None not in labels_list, "存在未加载的标签"

    # 转换为张量
    data_tensor = torch.stack(data_list)
    labels_tensor = torch.LongTensor(labels_list)

    return data_tensor, labels_tensor, corruption


def prepare_imagenet_c_dataloader(args, partition_idxs, is_train, mode="iid"):
    """辅助函数：准备DataLoader"""
    data_tensor, labels_tensor, corruption = make_imagenet_c(
        partition_idxs=partition_idxs,
        is_train=is_train,
        data_dir=args.data_dir,
        mode=mode
    )
    dataset = ImageNetCDataset(data_tensor, labels_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    return dataloader, corruption


def random_distortion(range_severity=(1, 6), mode='train'):
    if mode == "train" or mode == "iid":
        selected_distortions = distortions
    elif mode == "ood":
        selected_distortions = test_distortions
    else:
        raise NotImplementedError

    num_distortions = len(selected_distortions)
    i = np.random.randint(num_distortions)  # randomly choose a distortion
    s = np.random.randint(low=range_severity[0], high=range_severity[1])  # randomly choose severity from 1 to 5
    return selected_distortions[i], s


def test():
    partition_idxs = {
        0: [*range(1)],
        1: [*range(1, 2)]
    }
    is_train = {0: True, 1: False}
    make_imagenet_c(partition_idxs, is_train)


# if __name__ == "__main__":
#     test()