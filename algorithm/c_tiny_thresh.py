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
    from options import args_parser

    print("主模块（model/options）导入成功!")

    import utils

    print("utils模块导入成功!")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


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
    """加载Tiny-ImageNet预训练模型，兼容多种格式"""
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


def calculate_and_save_class_margins_tinyimagenet(device, dataset_path, model_path):
    """计算Tiny-ImageNet的类别边界阈值"""
    args = args_parser()
    # 强制设置类别数为200，确保与Tiny-ImageNet匹配
    args.num_classes = 200
    # 初始化模型并加载权重
    model = create_model(args)
    model.to(device)
    load_checkpoint(model, model_path, device)
    model.change_bn(mode='grad')
    model.eval()

    # Tiny-ImageNet专属数据预处理
    transform = transforms.Compose([
        transforms.Resize(64),  # Tiny-ImageNet图像尺寸为64x64
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  # Tiny-ImageNet标准化参数
    ])

    # 加载Tiny-ImageNet数据集（使用ImageFolder读取，其目录结构符合要求）
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, 'train'),
        transform=transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, 'val'),
        transform=transform
    )
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    print(f"成功加载Tiny-ImageNet数据集: {dataset_path}，共{len(full_dataset)}个样本")

    # 初始化边界统计容器
    class_margin_stats = defaultdict(lambda: [0.0, 0])
    num_classes = 200  # Tiny-ImageNet固定200个类别

    prefix = "_".join(args.newco.split("_")[:2])  # 例如 "digital_blur"

    # 遍历数据集计算边界阈值
    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader), desc="Processing Tiny-ImageNet"):
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

    # 整理统计结果并生成Tensor
    avg_margins = torch.zeros(num_classes, dtype=torch.float32)  # 创建Tensor存储平均值
    final_margin_stats = {}

    for class_id in range(num_classes):
        sum_margin, sample_count = class_margin_stats.get(class_id, [0.0, 0])
        avg_margin = sum_margin / sample_count if sample_count > 0 else 0.0
        final_margin_stats[class_id] = {
            "sum_margin": sum_margin,
            "sample_count": sample_count,
            "avg_margin": avg_margin
        }
        avg_margins[class_id] = avg_margin  # 存入Tensor

    # 保存PKL格式
    pkl_file_path = f'/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/class_thresh3/tinyimagenet/{prefix}_au4_2_176_0.25_margins_resnet18_tinyimagenet.pkl'
    # 创建文件夹（如果不存在）
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)

    # 保存为pkl格式
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(avg_margins, f)
    print(f"PKL格式结果已保存至: {pkl_file_path}")
    print(f"Tensor形状: {avg_margins.shape}, 数据类型: {avg_margins.dtype}")

    # 打印统计结果（前10类+最后10类）
    print(f"\n=== Tiny-ImageNet Class Margin Statistics (All Samples) ===")
    print("前10类统计：")
    for class_id in range(10):
        stats = final_margin_stats[class_id]
        print(
            f"Class {class_id:>3}: Sample Count={stats['sample_count']:>4}, Sum Margin={stats['sum_margin']:>8.4f}, Avg Margin={stats['avg_margin']:>6.4f}")
    print("...")
    print("最后10类统计：")
    for class_id in range(190, 200):
        stats = final_margin_stats[class_id]
        print(
            f"Class {class_id:>3}: Sample Count={stats['sample_count']:>4}, Sum Margin={stats['sum_margin']:>8.4f}, Avg Margin={stats['avg_margin']:>6.4f}")


def main():
    """主函数"""
    set_random_seed(42)
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"使用设备: {device}")

    # Tiny-ImageNet配置
    dataset_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/data/tiny-imagenet-200'  # 数据集根目录
    model_path = args.load_model_path
    # 执行类别边界统计
    calculate_and_save_class_margins_tinyimagenet(device, dataset_path, model_path)


if __name__ == '__main__':
    main()
