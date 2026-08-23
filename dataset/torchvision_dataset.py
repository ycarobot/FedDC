import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import scipy.io as sio  # 新增：导入scipy.io用于读取mat文件

def create_torchvision_dataset(dataset_name='mnist', data_dir='../data'):
    """
    使用torchvision中的数据集，仅修改ImageNet路径
    :param dataset_name: 数据集名称（mnist/fmnist/cifar10/cifar100/imagenet）
    :param data_dir: 数据集根目录，默认使用../data
    :return: train_dataset, test_dataset
    """
    base_data_dir = os.path.abspath(os.path.expanduser(data_dir))
    # torchvision-managed datasets share this cache directory.
    data_dir = os.path.join(base_data_dir, 'torchvision')

    if dataset_name == 'imagenet':
        # 单独设置ImageNet的训练集和验证集路径
        train_dir = os.path.join(base_data_dir, 'ImageNet_train')
        val_dir = os.path.join(base_data_dir, 'ImageNet_val', 'val')

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 使用ImageFolder加载自定义路径的ImageNet（原datasets.ImageNet要求特定目录结构）
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean and std of mnist
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))])  # mean and std of fmnist
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'food101':
        # Food-101 数据集：101 个食物类别，约 101,000 张图片
        # 使用 torchvision 内置的 Food101 数据集
        train_transform = transforms.Compose([
            transforms.Resize(256),  # 原始图片大小不一，先调整大小
            transforms.RandomCrop(224),  # 随机裁剪到 224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet 标准归一化
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),  # 调整大小
            transforms.CenterCrop(224),  # 中心裁剪到 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载 Food-101 数据集
        train_dataset = datasets.Food101(root=data_dir, split='train',
                                         download=True, transform=train_transform)
        test_dataset = datasets.Food101(root=data_dir, split='test',
                                        download=True, transform=test_transform)

        # print(f"Food-101训练集样本数: {len(train_dataset)}")
        # print(f"Food-101测试集样本数: {len(test_dataset)}")
        # print(f"Food-101类别数量: {len(train_dataset.classes)}")

    elif dataset_name == 'coarse-cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        train_dataset.targets = sparse2coarse(train_dataset.targets)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        test_dataset.targets = sparse2coarse(test_dataset.targets)

    elif dataset_name == 'cinic10':
        # CINIC-10数据路径（包含train/val/test三个子目录）
        cinic_dir = os.path.join(base_data_dir, 'CINIC-10')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                 (0.24205776, 0.23828046, 0.25874835)),
        ])

        # 加载训练集（包含train和val）
        train_dataset = datasets.ImageFolder(
            root=os.path.join(cinic_dir, 'train'),
            transform=transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(cinic_dir, 'valid'),
            transform=transform
        )

        # 合并train和val作为训练集
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

        # 测试集
        test_dataset = datasets.ImageFolder(
            root=os.path.join(cinic_dir, 'test'),
            transform=transform
        )

        # print(f"CINIC-10训练集样本数: {len(train_dataset)}")
        # print(f"CINIC-10测试集样本数: {len(test_dataset)}")

    elif dataset_name == 'tiny_imagenet':
        # TinyImageNet路径设置
        tiny_dir = os.path.join(base_data_dir, 'Tiny-ImageNet')

        # 定义数据转换（使用TinyImageNet专用均值和标准差）
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2302, 0.2265, 0.2262])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2302, 0.2265, 0.2262])
        ])

        # 使用自定义Dataset类加载
        train_dataset = TinyImageNetDataset(
            root=os.path.join(tiny_dir, 'train'),
            transform=train_transform
        )
        test_dataset = TinyImageNetDataset(
            root=os.path.join(tiny_dir, 'val'),
            transform=test_transform,
            is_train=False
        )






    elif dataset_name == 'stanfordcars':

        # 适配你的Stanford Cars数据集实际路径

        cars_dir = os.path.join(base_data_dir, 'Stanford_Cars')

        # 检查关键文件是否存在

        mat_file_path = os.path.join(cars_dir, 'cars_annos.mat')

        img_dir = os.path.join(cars_dir, 'car_ims')

        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"未找到mat标注文件: {mat_file_path}")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"未找到图片目录: {img_dir}")

        # print(f"加载Stanford Cars数据集，mat文件: {mat_file_path}")
        #
        # print(f"图片目录: {img_dir}")

        # 数据增强和预处理

        train_transform = transforms.Compose([

            transforms.Resize((256, 256)),

            transforms.RandomCrop(224),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])

        ])

        test_transform = transforms.Compose([

            transforms.Resize((256, 256)),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])

        ])

        # 加载训练集和测试集

        train_dataset = StanfordCarsNestedDataset(

            data_dir=cars_dir,

            mat_file_path=mat_file_path,

            split='train',

            transform=train_transform

        )

        test_dataset = StanfordCarsNestedDataset(

            data_dir=cars_dir,

            mat_file_path=mat_file_path,

            split='test',

            transform=test_transform

        )

        # print(f"Stanford Cars训练集样本数: {len(train_dataset)}")
        #
        # print(f"Stanford Cars测试集样本数: {len(test_dataset)}")
        #
        # print(f"Stanford Cars类别数量: {train_dataset.num_classes}")
        #
        # # 显示一些统计信息
        #
        # if len(train_dataset) > 0:
        #
        #     print(f"训练集示例: 图片路径={train_dataset.images[0]['path']}, 标签={train_dataset.labels[0]}")
        #
        #     if hasattr(train_dataset, 'class_names') and train_dataset.labels[0] < len(train_dataset.class_names):
        #         class_name = train_dataset.class_names[train_dataset.labels[0] - 1]  # 标签从1开始，列表从0开始
        #
        #         print(f"类别名称: {class_name}")


    else:

        raise NotImplementedError(f'不支持的数据集：{dataset_name}！请检查数据集名称。')

    return train_dataset, test_dataset






class TinyImageNetDataset(Dataset):
    """TinyImageNet自定义数据集类"""

    def __init__(self, root, transform=None, is_train=True):
        self.root = root
        self.transform = transform
        self.is_train = is_train

        # 加载wnids.txt建立类别映射
        self.id_dict = {}
        wnids_path = os.path.join(os.path.dirname(root), 'wnids.txt')
        with open(wnids_path, 'r') as f:
            for i, line in enumerate(f):
                self.id_dict[line.strip()] = i

        # 加载图像路径和标签
        self.images = []
        self.labels = []

        if is_train:
            # 训练集: root/class/images/*.JPEG
            for class_name in os.listdir(root):
                class_dir = os.path.join(root, class_name, 'images')
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.id_dict[class_name])
        else:
            # 验证集: root/images/*.JPEG + val_annotations.txt
            val_annotations = os.path.join(root, 'val_annotations.txt')
            img_to_class = {}

            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name, class_name = parts[0], parts[1]
                        img_to_class[img_name] = self.id_dict.get(class_name, -1)

            for img_name in os.listdir(os.path.join(root, 'images')):
                if img_name.endswith('.JPEG') and img_name in img_to_class:
                    self.images.append(os.path.join(root, 'images', img_name))
                    self.labels.append(img_to_class[img_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class StanfordCarsNestedDataset(Dataset):
    """适配你的Stanford Cars数据集路径的自定义Dataset类"""

    def __init__(self, data_dir, split='train', transform=None, mat_file_path=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        # 优先使用传入的 mat_file_path，否则使用默认路径
        if mat_file_path and os.path.exists(mat_file_path):
            self.annos_path = mat_file_path
        else:
            self.annos_path = os.path.join(data_dir, 'cars_annos.mat')

        # 你的图片根目录：car_ims
        self.img_root = os.path.join(data_dir, 'car_ims')

        self.images = []
        self.labels = []
        self.num_classes = 0
        self.class_names = []

        # 加载数据
        self._load_data()

    def _load_data(self):
        #print(f"加载Stanford Cars标注文件: {self.annos_path}")

        # 加载mat文件
        try:
            mat_data = sio.loadmat(self.annos_path)
            # 兼容不同的mat文件结构
            if 'annotations' in mat_data:
                annos = mat_data['annotations'][0]  # 获取标注数组
            elif 'annotation' in mat_data:
                annos = mat_data['annotation'][0]
            else:
                raise KeyError("mat文件中未找到 'annotations' 或 'annotation' 字段")

            # 加载类别名称
            if 'class_names' in mat_data:
                self.class_names = [name[0] for name in mat_data['class_names'][0]]
            elif 'classname' in mat_data:
                self.class_names = [name[0] for name in mat_data['classname'][0]]
        except Exception as e:
            raise RuntimeError(f"加载mat文件失败: {e}")

        # 区分训练/测试集（0=训练集，1=测试集）
        split_flag = 0 if self.split == 'train' else 1
        #print(f"正在加载 {self.split} 集数据...")

        valid_count = 0
        error_count = 0

        for idx, anno in enumerate(annos):
            try:
                # 解析mat文件字段
                img_path_in_mat = anno[0][0] if len(anno[0]) > 0 else ''  # mat中存储的相对路径
                is_test = int(anno[6][0][0]) if len(anno[6]) > 0 else 0

                # 只处理当前split的数据
                if is_test != split_flag:
                    continue

                # 构建实际图片路径（拼接car_ims目录）
                # 例如：mat中的路径是 'car_ims/00001.jpg'，直接拼接即可
                full_img_path = os.path.join(self.data_dir, img_path_in_mat)

                # 兼容mat中路径不带car_ims前缀的情况
                if not os.path.exists(full_img_path):
                    full_img_path = os.path.join(self.img_root, os.path.basename(img_path_in_mat))

                # 检查文件是否存在
                if not os.path.exists(full_img_path):
                    print(f"警告：文件不存在 {full_img_path}")
                    error_count += 1
                    continue

                # 解析标签（Stanford Cars标签从1开始）
                label = int(anno[5][0][0]) if len(anno[5]) > 0 else 0
                # 可选：解析边界框（如果需要使用）
                bbox_x1 = float(anno[1][0][0]) if len(anno[1]) > 0 else 0
                bbox_y1 = float(anno[2][0][0]) if len(anno[2]) > 0 else 0
                bbox_x2 = float(anno[3][0][0]) if len(anno[3]) > 0 else 0
                bbox_y2 = float(anno[4][0][0]) if len(anno[4]) > 0 else 0

                # 添加到数据集
                self.images.append({
                    'path': full_img_path,
                    'bbox': (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                })
                self.labels.append(label)  # 保持原始标签（从1开始）
                valid_count += 1

            except Exception as e:
                print(f"解析第{idx}个标注时出错: {e}")
                error_count += 1
                continue

        # 统计类别数
        unique_labels = set(self.labels)
        self.num_classes = len(unique_labels)
        # print(f"找到 {self.num_classes} 个类别")
        # print(f"{self.split}集: 成功加载 {valid_count} 张图片，失败 {error_count} 张")

        # 检查是否加载到数据
        if valid_count == 0:
            raise RuntimeError(f"未找到任何{self.split}集的图片，请检查路径是否正确")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图片
        img_info = self.images[idx]
        try:
            img = Image.open(img_info['path']).convert('RGB')
        except Exception as e:
            print(f"加载图片失败 {img_info['path']}: {e}")
            # 返回空白图片和无效标签
            img = Image.new('RGB', (224, 224), color='black')

        label = self.labels[idx] - 1  # 标签转换为从0开始（适配PyTorch训练）

        # 应用变换
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
def sparse2coarse(targets):
    """将Pytorch CIFAR100的细粒度标签转换为粗粒度标签"""
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


def test():
    import torch
    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # 测试加载ImageNet数据集
    train_dataset, test_dataset = create_torchvision_dataset(dataset_name='imagenet')
    print(f"ImageNet训练集样本数: {len(train_dataset)}")
    print(f"ImageNet测试集样本数: {len(test_dataset)}")

    # 查看样本结构（仅显示前3个样本）
    for i in range(3):
        img, label = train_dataset[i]
        print(f"样本{i}形状: {img.shape}, 标签: {label}")

    # 可视化第一个样本
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"ImageNet样本，标签: {label}")
    plt.show()


# if __name__ == "__main__":
#     test()
