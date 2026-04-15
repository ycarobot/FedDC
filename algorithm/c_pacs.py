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

# 关键修复：将所有可能需要的目录添加到Python路径
sys.path.append(src_dir)  # 添加src目录
sys.path.append(utils_dir)  # 直接添加utils目录
sys.path.append(project_root)  # 添加项目根目录

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
    # 从src目录下直接导入
    from model.create_model import create_model
    from options_p5 import args_parser

    print("主模块导入成功!")

    # 测试导入utils模0块
    import utils

    print("utils模块导入成功!")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_and_save_class_margins_pacs(device, domain_name, dataset_path):
    args = args_parser()
    model = create_model(args)
    model.change_bn(mode='grad')
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domain_path = os.path.join(dataset_path, domain_name)
    domain_dataset = torchvision.datasets.ImageFolder(root=domain_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        domain_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    class_margin_stats = defaultdict(lambda: [0.0, 0])
    num_classes = 7

    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader), desc=f"Processing {domain_name}"):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            logits = model(X)
            probs = F.softmax(logits, dim=1)

            probs_sorted, _ = torch.sort(probs, descending=True, dim=1)
            margins = probs_sorted[:, 0] - probs_sorted[:, 1]

            for idx in range(len(Y)):
                class_id = Y[idx].item()
                margin_val = margins[idx].item()
                class_margin_stats[class_id][0] += margin_val
                class_margin_stats[class_id][1] += 1

    final_margin_stats = {}
    for class_id in range(num_classes):
        sum_margin, sample_count = class_margin_stats.get(class_id, [0.0, 0])
        avg_margin = sum_margin / sample_count if sample_count > 0 else 0.0
        final_margin_stats[class_id] = {
            "sum_margin": sum_margin,
            "sample_count": sample_count,
            "avg_margin": avg_margin
        }

    metrics_file_path = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/class_thresh1/pacs_aug/avg_class_margins_0.1init_domain_res18_test_PACS_{}.pkl'.format(
        domain_name)
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    with open(metrics_file_path, 'wb') as f:
        pickle.dump(final_margin_stats, f)

    print(f"\n=== Domain {domain_name} Class Margin Statistics (All Samples) ===")
    for class_id in range(num_classes):
        stats = final_margin_stats[class_id]
        print(f"Class {class_id}: "
              f"Sample Count={stats['sample_count']:>4}, "
              f"Sum Margin={stats['sum_margin']:>8.4f}, "
              f"Avg Margin={stats['avg_margin']:>6.4f}")
    print(f"\nResults saved to: {metrics_file_path}")

    return final_margin_stats


def calculate_average_thresholds(device, exclude_domain, dataset_path):
    """
    计算除指定域外的三个域的阈值，并求各类的平均值
    """
    # 所有域
    all_domains = ['art_painting', 'cartoon', 'photo', 'sketch']

    # 确定要计算的域（排除指定域）
    domains_to_calculate = [domain for domain in all_domains if domain != exclude_domain]

    print(f"排除的域: {exclude_domain}")
    print(f"要计算的域: {domains_to_calculate}")

    # 存储每个域的阈值统计
    domain_margin_stats = {}

    # 计算每个域的阈值
    for domain in domains_to_calculate:
        print(f"\n{'=' * 60}")
        margin_stats = calculate_and_save_class_margins_pacs(device, domain, dataset_path)
        domain_margin_stats[domain] = margin_stats

    # 计算各类的平均阈值
    num_classes = 7
    class_avg_thresholds = {}
    threshold_tensor = torch.zeros(num_classes, dtype=torch.float32)

    for class_id in range(num_classes):
        class_margins = []

        for domain in domains_to_calculate:
            if class_id in domain_margin_stats[domain]:
                avg_margin = domain_margin_stats[domain][class_id]['avg_margin']
                class_margins.append(avg_margin)

        if class_margins:  # 确保有数据
            # 计算平均值
            avg_margin = np.mean(class_margins)
            threshold_tensor[class_id] = avg_margin

            class_avg_thresholds[class_id] = {
                'avg_margin': avg_margin,
                'domain_margins': {domain: domain_margin_stats[domain][class_id]['avg_margin']
                                   for domain in domains_to_calculate}
            }
        else:
            # 如果没有数据，设置为0
            threshold_tensor[class_id] = 0.0

    # 保存为pickle格式，但内容为Tensor（后缀.pkl，内容为Tensor）
    tensor_thresholds_file = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/class_thresh1/pacs_aug/avg_class_thresholds_0.1init_domain_res18_test_exclude_{}.pkl'.format(
        exclude_domain)
    os.makedirs(os.path.dirname(tensor_thresholds_file), exist_ok=True)

    # 将Tensor保存为pickle文件
    with open(tensor_thresholds_file, 'wb') as f:
        pickle.dump(threshold_tensor, f)

    # 打印结果
    print(f"\n{'=' * 80}")
    print(f"排除域 '{exclude_domain}' 后的各类平均阈值")
    print(f"{'=' * 80}")
    for class_id in range(num_classes):
        if class_id in class_avg_thresholds:
            thresholds = class_avg_thresholds[class_id]
            print(f"Class {class_id}: "
                  f"各域阈值={thresholds['domain_margins']}, "
                  f"平均阈值={thresholds['avg_margin']:>6.4f}")

    print(f"\nTensor格式阈值: {threshold_tensor}")
    print(f"\nTensor阈值已保存为pkl文件: {tensor_thresholds_file}")

    return class_avg_thresholds, threshold_tensor

#all_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
def main():
    set_random_seed(42)
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    dataset_path = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/data/PACS'

    # 排除指定的域并计算平均阈值
    exclude_domain = 'art_painting'  # 可以修改为其他域
    class_avg_thresholds, threshold_tensor = calculate_average_thresholds(device, exclude_domain, dataset_path)

    # 打印最终的Tensor格式阈值
    print(f"\n{'=' * 80}")
    print(f"最终Tensor格式阈值 (排除域 '{exclude_domain}'):")
    print(f"{'=' * 80}")
    for class_id in range(len(threshold_tensor)):
        print(f"Class {class_id}: {threshold_tensor[class_id]:.6f}")
    print(f"\n完整Tensor: {threshold_tensor}")

    # 验证保存的Tensor文件
    print(f"\n验证保存的Tensor文件...")
    with open(
            '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/ATP-master/src/class_thresh1/pacs_aug/avg_class_thresholds_0.1init_domain_res18_test_exclude_{}.pkl'.format(
                    exclude_domain), 'rb') as f:
        loaded_tensor = pickle.load(f)
    print(f"从pkl文件加载的Tensor: {loaded_tensor}")
    print(f"Tensor类型: {type(loaded_tensor)}")
    print(f"Tensor形状: {loaded_tensor.shape}")


if __name__ == '__main__':
    main()