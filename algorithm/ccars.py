import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from collections import defaultdict
import pickle
from tqdm import tqdm
import random
import numpy as np
import scipy.io as sio
from PIL import Image
import sys

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
print(f"当前文件路径: {current_file}")

# 计算各层级目录路径
src_dir = os.path.dirname(os.path.dirname(current_file))  # src目录
utils_dir = os.path.join(src_dir, 'utils')  # utils目录
project_root = os.path.dirname(src_dir)  # 项目根目录

# 添加目录到Python路径
sys.path.append(src_dir)
sys.path.append(utils_dir)
sys.path.append(project_root)

# 打印路径信息用于调试
print(f"src目录路径: {src_dir}")
print(f"utils目录路径: {utils_dir}")
print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path}")

# 验证utils目录内容
if os.path.exists(utils_dir):
    print(f"utils目录内容: {os.listdir(utils_dir)}")
else:
    print(f"utils目录不存在: {utils_dir}")

# 尝试导入模块
try:
    from model.create_model import create_model
    from options_s1 import args_parser

    print("主模块（model/options）导入成功!")

    import utils

    print("utils模块导入成功!")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 导入你项目中的数据集加载函数（优先复用项目统一逻辑）
try:
    from data_loader import create_torchvision_dataset  # 替换为你实际的模块名
except ImportError:
    # 如果无法直接导入，内置实现和你项目中完全一致的StanfordCars加载逻辑
    print("警告：未找到create_torchvision_dataset函数，使用内置StanfordCars实现")


    class StanfordCarsNestedDataset(Dataset):
        """和你项目中完全一致的Stanford Cars自定义Dataset类"""

        def __init__(self, data_dir, split='train', transform=None, mat_file_path=None):
            self.data_dir = data_dir
            self.split = split
            self.transform = transform
            # 优先使用传入的 mat_file_path，否则使用默认路径
            if mat_file_path and os.path.exists(mat_file_path):
                self.annos_path = mat_file_path
            else:
                self.annos_path = os.path.join(data_dir, 'cars_annos.mat')

            # 图片根目录：car_ims
            self.img_root = os.path.join(data_dir, 'car_ims')

            self.images = []
            self.labels = []
            self.num_classes = 0
            self.class_names = []

            # 加载数据
            self._load_data()

        def _load_data(self):
            print(f"加载Stanford Cars标注文件: {self.annos_path}")

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
            print(f"正在加载 {self.split} 集数据...")

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

                    # 添加到数据集
                    self.images.append({
                        'path': full_img_path,
                        'bbox': (anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0][0]) if len(anno) >= 5 else (
                        0, 0, 0, 0)
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
            print(f"找到 {self.num_classes} 个类别")
            print(f"{self.split}集: 成功加载 {valid_count} 张图片，失败 {error_count} 张")

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


    def create_torchvision_dataset_stanfordcars(data_dir='../data'):
        """和你项目中完全一致的StanfordCars加载逻辑"""
        # 适配你的Stanford Cars数据集实际路径
        cars_dir = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/data/atp'

        # 检查关键文件是否存在
        mat_file_path = os.path.join(cars_dir, 'cars_annos.mat')
        img_dir = os.path.join(cars_dir, 'car_ims')

        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"未找到mat标注文件: {mat_file_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"未找到图片目录: {img_dir}")

        print(f"加载Stanford Cars数据集，mat文件: {mat_file_path}")
        print(f"图片目录: {img_dir}")

        # 数据增强和预处理（完全复用你项目中的配置）
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

        print(f"Stanford Cars训练集样本数: {len(train_dataset)}")
        print(f"Stanford Cars测试集样本数: {len(test_dataset)}")
        print(f"Stanford Cars类别数量: {train_dataset.num_classes}")

        return train_dataset, test_dataset


def set_random_seed(seed):
    """固定随机种子，确保结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model, model_path, device):
    """加载Stanford Cars预训练模型，兼容多种格式"""
    try:
        # 优先用torch.load加载
        checkpoint = torch.load(model_path, map_location=device)
        print(f"成功用torch.load加载模型: {model_path}")
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Checkpoint格式错误（非字典）: {type(checkpoint)}")
    except Exception as e1:
        print(f"torch.load失败: {e1}，尝试用pickle加载")
        try:
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"成功用pickle加载模型: {model_path}")
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Pickle Checkpoint格式错误（非字典）: {type(checkpoint)}")
        except Exception as e2:
            print(f"pickle加载也失败: {e2}")
            sys.exit(1)


def calculate_and_save_class_margins_stanfordcars(device, data_dir, model_path):
    """计算Stanford Cars的类别边界阈值（完全复用项目统一的数据集加载逻辑）"""
    args = args_parser()

    # 初始化模型并加载权重
    model = create_model(args)
    model.to(device)
    load_checkpoint(model, model_path, device)
    model.change_bn(mode='grad')
    model.eval()

    # 核心：使用你项目中统一的StanfordCars加载方式
    try:
        # 方式1：直接调用项目中的create_torchvision_dataset
        train_dataset, test_dataset = create_torchvision_dataset(dataset_name='stanfordcars', data_dir=data_dir)
        # 获取类别数（从数据集提取）
        num_classes = train_dataset.num_classes
    except:
        # 方式2：使用内置的相同逻辑加载（备用方案）
        train_dataset, test_dataset = create_torchvision_dataset_stanfordcars(data_dir=data_dir)
        num_classes = train_dataset.num_classes

    # 更新模型类别数配置
    args.num_classes = num_classes
    print(f"设置模型类别数为: {num_classes}")

    # 合并训练集和测试集用于计算边界
    full_dataset = ConcatDataset([train_dataset, test_dataset])

    # 打印数据集信息
    print(f"成功加载Stanford Cars数据集: {data_dir}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"类别数量: {num_classes}")

    # 数据加载器配置（适配Stanford Cars大尺寸图片）
    dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=32,  # 汽车数据集图片尺寸大，减小batch_size避免显存溢出
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )

    # 初始化边界统计容器
    class_margin_stats = defaultdict(lambda: [0.0, 0])

    prefix = "_".join(args.newco.split("_")[:2]) if hasattr(args, 'newco') and args.newco else "default"

    # 遍历数据集计算边界阈值
    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader), desc="Processing Stanford Cars"):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            # 模型预测
            logits = model(X)
            probs = F.softmax(logits, dim=1)

            # 计算边界阈值：最大概率 - 第二大概率
            probs_sorted, _ = torch.sort(probs, descending=True, dim=1)
            margins = probs_sorted[:, 0] - probs_sorted[:, 1]

            # 按真实类别统计边界
            for idx in range(len(Y)):
                class_id = Y[idx].item()
                margin_val = margins[idx].item()
                class_margin_stats[class_id][0] += margin_val
                class_margin_stats[class_id][1] += 1

    # 整理统计结果
    final_margin_stats = {}
    avg_margins = []
    for class_id in range(num_classes):
        sum_margin, sample_count = class_margin_stats.get(class_id, [0.0, 0])
        avg_margin = sum_margin / sample_count if sample_count > 0 else 0.0
        final_margin_stats[class_id] = {
            "sum_margin": sum_margin,
            "sample_count": sample_count,
            "avg_margin": avg_margin
        }
        avg_margins.append(avg_margin)

    # 转换为Tensor并保存
    margins_tensor = torch.tensor(avg_margins, dtype=torch.float32)
    tensor_save_dir = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/class_thresh3/stanfordcars'
    os.makedirs(tensor_save_dir, exist_ok=True)
    tensor_file_path = os.path.join(tensor_save_dir,
                                    f'{prefix}_au4_2_51_0.25init_class_margins_resnet18_stanfordcars.pkl')

    with open(tensor_file_path, 'wb') as f:
        pickle.dump(margins_tensor, f)
    print(f"Tensor格式结果已保存至: {tensor_file_path}")
    print(f"Tensor形状: {margins_tensor.shape}, 数据类型: {margins_tensor.dtype}")

    # 打印统计结果
    print(f"\n=== Stanford Cars Class Margin Statistics (All Samples) ===")
    print("前10类统计：")
    for class_id in range(min(10, num_classes)):
        stats = final_margin_stats[class_id]
        print(
            f"Class {class_id:>3}: Sample Count={stats['sample_count']:>4}, Sum Margin={stats['sum_margin']:>8.4f}, Avg Margin={stats['avg_margin']:>6.4f}")
    print("...")
    print("最后10类统计：")
    for class_id in range(max(0, num_classes - 10), num_classes):
        stats = final_margin_stats[class_id]
        print(
            f"Class {class_id:>3}: Sample Count={stats['sample_count']:>4}, Sum Margin={stats['sum_margin']:>8.4f}, Avg Margin={stats['avg_margin']:>6.4f}")


def main():
    """主函数"""
    set_random_seed(42)
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"使用设备: {device}")

    # Stanford Cars配置（和你项目中一致的路径）
    data_dir = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/data'
    model_path = args.load_model_path if hasattr(args, 'load_model_path') else ''

    if not model_path:
        print("错误：未指定模型路径！请通过args.load_model_path设置")
        sys.exit(1)

    # 执行类别边界统计
    calculate_and_save_class_margins_stanfordcars(device, data_dir, model_path)


if __name__ == '__main__':
    main()