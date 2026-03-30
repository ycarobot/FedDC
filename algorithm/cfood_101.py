import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
import pickle
from tqdm import tqdm
import random
import numpy as np
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
    from options_f1 import args_parser

    print("主模块（model/options）导入成功!")

    import utils

    print("utils模块导入成功!")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 导入你项目中的数据集加载函数（核心修改：复用项目统一的加载逻辑）
# 确保这个函数所在的文件路径已被添加到sys.path
try:
    from data_loader import create_torchvision_dataset  # 替换为你实际的模块名
except ImportError:
    # 如果无法直接导入，就在当前文件中实现相同的food101加载逻辑
    print("警告：未找到create_torchvision_dataset函数，使用内置实现")


    def create_torchvision_dataset_food101(data_dir='../data'):
        """和你项目中完全一致的Food101加载逻辑"""
        data_dir = os.path.join(data_dir, 'torchvision')

        # 完全复用你提供的transform配置
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

        # 完全复用你提供的数据集加载参数
        train_dataset = torchvision.datasets.Food101(root=data_dir, split='train',
                                                     download=True, transform=train_transform)
        test_dataset = torchvision.datasets.Food101(root=data_dir, split='test',
                                                    download=True, transform=test_transform)

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
    """加载Food-101预训练模型，兼容多种格式"""
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


def calculate_and_save_class_margins_food101(device, data_dir, model_path):
    """计算Food-101的类别边界阈值（完全复用项目统一的数据集加载逻辑）"""
    args = args_parser()
    # 强制设置类别数为101，确保与Food-101匹配
    args.num_classes = 101
    # 初始化模型并加载权重
    model = create_model(args)
    model.to(device)
    load_checkpoint(model, model_path, device)
    model.change_bn(mode='grad')
    model.eval()

    # 核心修改：使用你项目中统一的数据集加载方式
    try:
        # 方式1：直接调用项目中的create_torchvision_dataset
        train_dataset, test_dataset = create_torchvision_dataset(dataset_name='food101', data_dir=data_dir)
    except:
        # 方式2：使用内置的相同逻辑加载（备用方案）
        train_dataset, test_dataset = create_torchvision_dataset_food101(data_dir=data_dir)

    # 合并训练集和测试集用于计算边界
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    # 打印数据集信息
    print(f"成功加载Food-101数据集: {data_dir}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"类别数量: {len(train_dataset.classes)}")

    # 数据加载器配置（适配Food-101大尺寸图片）
    dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=64,  # 适配大尺寸图片的batch_size
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )

    # 初始化边界统计容器
    class_margin_stats = defaultdict(lambda: [0.0, 0])
    num_classes = 101  # Food-101固定101个类别

    prefix = "_".join(args.newco.split("_")[:2]) if hasattr(args, 'newco') and args.newco else "default"

    # 遍历数据集计算边界阈值
    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader), desc="Processing Food-101"):
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
    tensor_save_dir = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/class_thresh3/food101'
    os.makedirs(tensor_save_dir, exist_ok=True)
    tensor_file_path = os.path.join(tensor_save_dir, f'{prefix}_au4_2_51_0.25init_class_margins_resnet18_food101.pkl')

    with open(tensor_file_path, 'wb') as f:
        pickle.dump(margins_tensor, f)
    print(f"Tensor格式结果已保存至: {tensor_file_path}")
    print(f"Tensor形状: {margins_tensor.shape}, 数据类型: {margins_tensor.dtype}")

    # 打印统计结果
    print(f"\n=== Food-101 Class Margin Statistics (All Samples) ===")
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

    # Food-101配置（和你项目中一致的路径）
    data_dir = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/data'  # 根目录，会自动拼接torchvision
    model_path = args.load_model_path if hasattr(args, 'load_model_path') else ''

    if not model_path:
        print("错误：未指定模型路径！请通过args.load_model_path设置")
        sys.exit(1)

    # 执行类别边界统计
    calculate_and_save_class_margins_food101(device, data_dir, model_path)


if __name__ == '__main__':
    main()