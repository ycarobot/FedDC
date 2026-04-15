import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os

from .distortions import distortions, test_distortions


def make_cifar10_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of CIFAR-10-C
        partition_idxs: dictionary of (client index : list of sample indices)
        is_train: whether each client is source client or target client
        data_dir: root of torchvision dataset
        mode: 'iid' or 'ood'.
            'iid' uses the same 15 distortions for both source and target clients
            'ood' uses 15 distortions for source client and 4 additional distortions for target clients
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_dataset, test_dataset])

    cifar_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            cifar_c[sid] = x
            labels[sid] = y
    assert None not in cifar_c
    assert None not in labels

    cifar_c = torch.stack(cifar_c)
    labels = torch.LongTensor(labels)

    return cifar_c, labels, corruption


def make_cifar100_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of CIFAR-100-C
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # mean and std of each channel
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_dataset, test_dataset])

    cifar_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            cifar_c[sid] = x
            labels[sid] = y
    assert None not in cifar_c
    assert None not in labels

    cifar_c = torch.stack(cifar_c)
    labels = torch.LongTensor(labels)

    return cifar_c, labels, corruption


def make_cinic10_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of CINIC-10-C
        partition_idxs: dictionary of (client index : list of sample indices)
        is_train: whether each client is source client or target client
        data_dir: root of torchvision dataset
        mode: 'iid' or 'ood'.
            'iid' uses the same 15 distortions for both source and target clients
            'ood' uses 15 distortions for source client and 4 additional distortions for target clients
    """
    data_dir = os.path.join(data_dir, 'CINIC-10')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])

    # Load all three parts of CINIC-10 dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Combine all three parts
    dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    cinic_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            cinic_c[sid] = x
            labels[sid] = y

    assert None not in cinic_c
    assert None not in labels

    cinic_c = torch.stack(cinic_c)
    labels = torch.LongTensor(labels)

    return cinic_c, labels, corruption


def make_food101_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of Food101 (无图像损坏版本)
        partition_idxs: dictionary of (client index : list of sample indices)
        is_train: whether each client is source client or target client (接口兼容，无实际作用)
        data_dir: root of Food101 dataset
        mode: 'iid'/'ood' (接口兼容，无实际作用)
    """
    data_dir = os.path.join(data_dir, 'food-101')

    # 预处理流程（无distortion，仅基础转换+归一化）
    # Food101标准归一化参数（来自torchvision官方推荐）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Food101图像尺寸不一，统一缩放到224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载Food101训练和测试集
    train_dataset = datasets.Food101(root=data_dir, split='train', download=True, transform=transform)
    test_dataset = datasets.Food101(root=data_dir, split='test', download=True, transform=transform)

    # 合并训练和测试集（与其他函数保持一致的处理方式）
    dataset = ConcatDataset([train_dataset, test_dataset])

    # 初始化数据存储容器
    food101_data, labels = [None] * len(dataset), [None] * len(dataset)

    # # 保留corruption字典接口（但值为空，因为无损坏）
    # corruption = {}
    #
    # # 按客户端索引分配数据
    # for cid, sids in tqdm(partition_idxs.items()):
    #     # 接口兼容：无损坏，所以corruption值设为('none', 0)
    #     corruption[cid] = ('none', 0)
    #
    #     for sid in sids:
    #         # 边界检查：防止索引越界
    #         if sid >= len(dataset):
    #             raise IndexError(f"Sample index {sid} exceeds Food101 dataset size {len(dataset)}")
    #
    #         x, y = dataset[sid]
    #         # 无distortion处理，直接存储
    #         food101_data[sid] = x
    #         labels[sid] = y

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = transform(x)
            food101_data[sid] = x
            labels[sid] = y

    # 验证数据完整性
    assert None not in food101_data, "存在未赋值的图像数据"
    assert None not in labels, "存在未赋值的标签数据"

    # 转换为tensor格式
    food101_data = torch.stack(food101_data)
    labels = torch.LongTensor(labels)

    return food101_data, labels, corruption


def make_stanfordcars_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of Stanford Cars dataset
        partition_idxs: dictionary of (client index : list of sample indices)
        is_train: whether each client is source client or target client
        data_dir: root of Stanford Cars dataset
        mode: 'iid' or 'ood'.
            'iid' uses the same 15 distortions for both source and target clients
            'ood' uses 15 distortions for source client and 4 additional distortions for target clients
    """
    # 设置Stanford Cars数据集路径
    stanford_cars_path = os.path.join(data_dir, 'Stanford_Cars')

    # 如果指定的路径不存在，使用绝对路径
    if not os.path.exists(stanford_cars_path):
        stanford_cars_path = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/data/Stanford_Cars'

    # 检查路径是否存在
    if not os.path.exists(stanford_cars_path):
        raise FileNotFoundError(f"Stanford Cars dataset not found at: {stanford_cars_path}")

    # 数据预处理流程
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Stanford Cars统一缩放到224x224
        transforms.ToTensor(),
        transforms.ToPILImage(),  # 转换为PIL图像用于后续distortion处理
    ])

    # 后处理流程（包含归一化）
    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # distortion处理后需要转换回PIL
        transforms.ToTensor(),
        # Stanford Cars的归一化参数（使用ImageNet标准参数）
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载Stanford Cars数据集
    # 注意：Stanford Cars在torchvision中是通过datasets.StanfordCars加载的
    # 但如果您有自定义路径，可以使用ImageFolder方式加载

    # 方法1：使用ImageFolder（如果数据集是按类别组织的）
    train_path = os.path.join(stanford_cars_path, 'train')
    test_path = os.path.join(stanford_cars_path, 'test')

    if os.path.exists(train_path) and os.path.exists(test_path):
        # 使用自定义路径的ImageFolder加载方式
        train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    else:
        # 方法2：使用torchvision的StanfordCars数据集（需要下载）
        # 首先尝试从指定路径加载，如果不存在则下载
        try:
            train_dataset = datasets.StanfordCars(
                root=data_dir,
                split='train',
                download=True,
                transform=transform
            )
            test_dataset = datasets.StanfordCars(
                root=data_dir,
                split='test',
                download=True,
                transform=transform
            )
        except AttributeError:
            # 如果torchvision版本不支持StanfordCars，使用自定义加载
            print("Warning: StanfordCars not available in this torchvision version.")
            print("Using ImageFolder with expected structure...")
            # 假设数据集结构为：Stanford_Cars/cars_train/ 和 Stanford_Cars/cars_test/
            train_path = os.path.join(stanford_cars_path, 'cars_train')
            test_path = os.path.join(stanford_cars_path, 'cars_test')
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
                test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
            else:
                raise FileNotFoundError(
                    f"Stanford Cars dataset not found. Checked paths:\n"
                    f"1. {stanford_cars_path}/train/\n"
                    f"2. {stanford_cars_path}/test/\n"
                    f"3. {stanford_cars_path}/cars_train/\n"
                    f"4. {stanford_cars_path}/cars_test/"
                )

    # 合并训练和测试集（与其他函数保持一致的处理方式）
    dataset = ConcatDataset([train_dataset, test_dataset])

    # 初始化数据存储容器
    stanford_cars_c, labels = [None] * len(dataset), [None] * len(dataset)

    # 记录每个客户端应用的distortion类型
    corruption = {}

    # 按客户端索引分配数据并应用distortion
    for cid, sids in tqdm(partition_idxs.items()):
        # 为每个客户端随机选择distortion类型和严重程度
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)

        # 记录该客户端的distortion信息
        corruption[cid] = (distortion.__name__, severity)

        # 处理该客户端的每个样本
        for sid in sids:
            # 边界检查
            if sid >= len(dataset):
                raise IndexError(f"Sample index {sid} exceeds Stanford Cars dataset size {len(dataset)}")

            # 获取原始图像和标签
            x, y = dataset[sid]

            # 应用distortion
            x = distortion(x, severity=severity)

            # 转换回uint8格式（与CIFAR处理方式一致）
            x = np.uint8(x)

            # 应用后处理（包括归一化）
            x = post_transform(x)

            # 存储处理后的图像和标签
            stanford_cars_c[sid] = x
            labels[sid] = y

    # 验证数据完整性
    assert None not in stanford_cars_c, "存在未赋值的Stanford Cars图像数据"
    assert None not in labels, "存在未赋值的Stanford Cars标签数据"

    # 转换为tensor格式
    stanford_cars_c = torch.stack(stanford_cars_c)
    labels = torch.LongTensor(labels)

    return stanford_cars_c, labels, corruption


# def make_imagenet_c(partition_idxs, is_train, data_dir='/mnt/sda/PythonProject/Learning_Dataset', mode="none"):
#     """
#     Load pre-generated ImageNet-C dataset from specific path
#     partition_idxs: dictionary of (client index : list of sample indices)
#     is_train: whether each client is source client or target client
#     data_dir: root directory (will use fixed ImageNet-C path)
#     mode: not used (kept for interface consistency)
#     """
#     # Fixed ImageNet-C path
#     imagenet_c_path = os.path.join(data_dir, 'ImageNet-C')
#
#     # All 15 standard ImageNet-C corruptions
#     CORRUPTIONS = [
#         'gaussian_noise', 'shot_noise', 'impulse_noise',
#         'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
#         'snow', 'frost', 'fog', 'brightness',
#         'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
#     ]
#
#     # Severity levels (1-5)
#     SEVERITIES = [1, 2, 3, 4, 5]
#
#     # Post-transform (standard ImageNet normalization)
#     post_transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     # Load clean ImageNet for original labels
#     clean_path = os.path.join(imagenet_c_path, 'clean')
#     clean_dataset = datasets.ImageFolder(clean_path, transform=transforms.ToTensor())
#
#     # Initialize containers
#     total_samples = len(clean_dataset)
#     corrupted_images = torch.zeros((total_samples, 3, 224, 224), dtype=torch.float32)
#     labels = torch.zeros(total_samples, dtype=torch.long)
#     corruption = {}
#
#     # Track which samples have been assigned
#     assigned_mask = np.zeros(total_samples, dtype=bool)
#
#     for cid, sids in tqdm(partition_idxs.items()):
#         # Randomly select corruption and severity
#         corruption_type = np.random.choice(CORRUPTIONS)
#         severity = np.random.choice(SEVERITIES)
#
#         # Load corrupted dataset
#         corrupt_path = os.path.join(imagenet_c_path, corruption_type, str(severity))
#         corrupt_dataset = datasets.ImageFolder(corrupt_path, transform=transforms.ToTensor())
#
#         # Store corruption info
#         corruption[cid] = (corruption_type, severity)
#
#         # Assign samples to client
#         for sid in sids:
#             if sid >= total_samples:
#                 raise IndexError(f"Sample index {sid} exceeds dataset size {total_samples}")
#
#             if assigned_mask[sid]:
#                 continue  # Skip already assigned samples
#
#             # Get corrupted image and original label
#             img, _ = corrupt_dataset[sid]  # Ignore folder-based label
#             original_label = clean_dataset[sid][1]
#
#             # Apply post-processing
#             img = post_transform(img)
#
#             # Store results
#             corrupted_images[sid] = img
#             labels[sid] = original_label
#             assigned_mask[sid] = True
#
#     # Verify all requested samples were assigned
#     if not np.all(assigned_mask[np.concatenate(list(partition_idxs.values()))]):
#         raise RuntimeError("Some requested samples were not assigned properly")
#
#     return corrupted_images, labels, corruption


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
    make_cifar10_c(partition_idxs)
