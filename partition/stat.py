import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ==================== 统一设置参数 ====================
# 设置期刊级别的绘图参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1.2

# 统一DPI设置
UNIFIED_DPI = 600

# 统一字体大小设置
UNIFIED_LABEL_FONTSIZE = 20
UNIFIED_TICK_FONTSIZE = 18
UNIFIED_TITLE_FONTSIZE = 22

# 统一显示范围设置
UNIFIED_CLASS_ID_RANGE = 100
UNIFIED_CLIENT_ID_RANGE = 300
# ==================== 参数设置结束 ====================


def create_heterogeneity_plot_from_partition(dataset, num_labels, partition_idxs, filename=None,
                                             max_clients_per_row=50,
                                             class_id_range=UNIFIED_CLASS_ID_RANGE,
                                             client_id_range=UNIFIED_CLIENT_ID_RANGE,
                                             dpi=UNIFIED_DPI,
                                             label_fontsize=UNIFIED_LABEL_FONTSIZE,
                                             tick_fontsize=UNIFIED_TICK_FONTSIZE):
    """
    从partition_idxs创建数据异构性可视化图 - 显示所有客户端

    参数:
        class_id_range: Class ID 的显示范围 (如 100, 200)
        client_id_range: Client ID 的显示范围 (如 300)
        dpi: 图片分辨率
        label_fontsize: 坐标轴标签字体大小
        tick_fontsize: 刻度标签字体大小
    """
    # 获取所有客户端
    all_client_ids = list(partition_idxs.keys())
    num_clients = len(all_client_ids)
    actual_num_labels = num_labels

    print(f"总客户端数: {num_clients}")
    print(f"实际类别数: {actual_num_labels}")
    print(f"Class ID 显示范围: 0-{class_id_range}")
    print(f"Client ID 显示范围: 0-{client_id_range}")
    print(f"图片DPI: {dpi}")

    # 计算标签分布
    labels = [data[-1] for data in dataset]
    label_dist = np.zeros((num_clients, actual_num_labels), dtype=int)

    for idx, cid in enumerate(all_client_ids):
        sids = partition_idxs[cid]
        for sid in sids:
            if 0 <= labels[sid] < actual_num_labels:
                label_dist[idx, labels[sid]] += 1

    print(f"总样本数: {label_dist.sum()}")
    print(f"每个客户端平均样本数: {label_dist.sum(axis=1).mean():.1f}")

    # 动态计算图形尺寸
    if num_clients <= max_clients_per_row:
        # 单行布局
        fig_width = 14
        fig_height = 10
        clients_per_row = num_clients
        num_rows = 1
    else:
        # 多行布局
        clients_per_row = max_clients_per_row
        num_rows = (num_clients + clients_per_row - 1) // clients_per_row
        fig_width = 16
        fig_height = 5 * num_rows

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_aspect('equal')

    # 设置背景
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')

    # 归一化分布
    row_sums = label_dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    label_dist_normalized = label_dist / row_sums

    # 使用颜色
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, actual_num_labels))

    # 绘制圆圈
    max_proportion = label_dist_normalized.max()
    if max_proportion == 0:
        max_proportion = 1

    # 动态调整圆圈大小因子
    if num_clients <= 20:
        circle_scale = 100
    elif num_clients <= 50:
        circle_scale = 0.8
    else:
        circle_scale = 0.6

    circle_count = 0

    # 计算坐标映射 - 数据索引从0开始，但显示坐标从1开始
    # 实际数据范围: 0到actual_num_labels-1, 0到num_clients-1
    # 显示坐标范围: 1到class_id_range, 1到client_id_range
    x_scale = (class_id_range - 1) / max(1, (actual_num_labels - 1))
    y_scale = (client_id_range - 1) / max(1, (num_clients - 1))

    for client_idx in range(num_clients):
        for label_idx in range(actual_num_labels):
            proportion = label_dist_normalized[client_idx, label_idx]

            if proportion > 0.001:
                circle_size = (proportion / max_proportion) * circle_scale * 100

                # 数据索引映射到显示坐标 - 从1开始
                if actual_num_labels > 1:
                    x_pos = 1 + label_idx * x_scale
                else:
                    x_pos = class_id_range / 2  # 只有一个类别时居中显示

                if num_clients > 1:
                    y_pos = 1 + client_idx * y_scale
                else:
                    y_pos = client_id_range / 2  # 只有一个客户端时居中显示

                circle = Circle((x_pos, y_pos),
                                radius=circle_size,
                                color=colors[label_idx],
                                alpha=0.8,
                                edgecolor='white',
                                linewidth=0.5)
                ax.add_patch(circle)
                circle_count += 1

    print(f"总共绘制了 {circle_count} 个圆圈")

    # 设置坐标轴范围 - 从0开始显示，但数据从1开始
    ax.set_xlim(-0.5, class_id_range + 0.5)
    ax.set_ylim(-0.5, client_id_range + 0.5)

    # 设置标签 - 使用传入的字体大小
    ax.set_xlabel('Class ID', fontsize=label_fontsize, labelpad=15)
    ax.set_ylabel('Client ID', fontsize=label_fontsize, labelpad=15)

    # 设置刻度 - 使用指定的显示范围，从0开始
    def calculate_ticks_interval(max_value):
        """根据最大值计算合适的刻度间隔"""
        if max_value <= 20:
            return 5
        elif max_value <= 50:
            return 10
        elif max_value <= 100:
            return 20
        elif max_value <= 200:
            return 40
        elif max_value <= 300:
            return 50
        elif max_value <= 500:
            return 100
        else:
            return max(50, max_value // 10)

    # X轴刻度 - 从0开始
    x_ticks_interval = calculate_ticks_interval(class_id_range)
    x_ticks = list(range(0, class_id_range + 1, x_ticks_interval))
    if class_id_range not in x_ticks:
        x_ticks.append(class_id_range)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=tick_fontsize)

    # Y轴刻度 - 从0开始
    y_ticks_interval = calculate_ticks_interval(client_id_range)
    y_ticks = list(range(0, client_id_range + 1, y_ticks_interval))
    if client_id_range not in y_ticks:
        y_ticks.append(client_id_range)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=tick_fontsize)

    # 添加网格 - 只在刻度位置显示网格
    ax.grid(True, alpha=0.35, linestyle='-', linewidth=0.4, color='#e0e0e0')

    # 设置坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    plt.tight_layout()

    # 保存图片
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white',
                    edgecolor='none', format='png')
        print(f"保存图片: {filename}")

    plt.show()

    return fig, ax, label_dist


def print_label_distribution_stat(dataset, num_labels, partition_idxs, visualize=False, resize=0.2,
                                  class_id_range=UNIFIED_CLASS_ID_RANGE,
                                  client_id_range=UNIFIED_CLIENT_ID_RANGE,
                                  dpi=UNIFIED_DPI,
                                  label_fontsize=UNIFIED_LABEL_FONTSIZE-2,
                                  tick_fontsize=UNIFIED_TICK_FONTSIZE-2):
    """
    打印标签分布统计信息
    """
    num_clients = len(partition_idxs)
    labels = [data[-1] for data in dataset]

    label_dist = np.zeros((num_clients, num_labels), dtype=int)
    for cid, sids in partition_idxs.items():
        for sid in sids:
            label_dist[cid, labels[sid]] += 1

    print('Label Distribution:')
    print(label_dist)

    if visualize:
        # 原有的散点图
        x = np.tile(np.arange(num_clients), num_labels)
        y = np.repeat(np.arange(num_labels), num_clients)
        size = label_dist[x, y]

        plt.figure(figsize=(12, 8), dpi=dpi)
        plt.scatter(x, y, s=size * resize, color='blue', alpha=0.7)

        # 调整散点图的刻度间隔
        if num_clients > 20:
            x_interval = max(1, num_clients // 10)
            x_ticks = list(range(0, num_clients, x_interval))
            if x_ticks[-1] < num_clients - 1:
                x_ticks.append(num_clients - 1)
            plt.xticks(x_ticks, fontsize=tick_fontsize)
        else:
            plt.xticks(range(num_clients), fontsize=tick_fontsize)

        if num_labels > 20:
            y_interval = max(1, num_labels // 10)
            y_ticks = list(range(0, num_labels, y_interval))
            if y_ticks[-1] < num_labels - 1:
                y_ticks.append(num_labels - 1)
            plt.yticks(y_ticks, fontsize=tick_fontsize)
        else:
            plt.yticks(range(num_labels), fontsize=tick_fontsize)

        plt.xlabel('Client ID', fontsize=label_fontsize)
        plt.ylabel('Label ID', fontsize=label_fontsize)
        plt.title('Hybrid Shift Label Distribution on Tiny-ImageNet', fontsize=label_fontsize + 2)
        plt.grid(True, alpha=0.3)
        plt.show()

        # 新增：创建异构性可视化图 - 所有客户端
        create_heterogeneity_plot_from_partition(
            dataset, num_labels, partition_idxs,
            class_id_range=class_id_range,
            client_id_range=client_id_range,
            dpi=dpi,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize
        )


def print_label_distribution_stat_office_home(dataset, num_labels, partition_idxs, visualize=False, resize=0.2,
                                              environment=0,
                                              class_id_range=UNIFIED_CLASS_ID_RANGE,
                                              client_id_range=UNIFIED_CLIENT_ID_RANGE,
                                              dpi=UNIFIED_DPI,
                                              label_fontsize=UNIFIED_LABEL_FONTSIZE-2,
                                              tick_fontsize=UNIFIED_TICK_FONTSIZE-2):
    """
    Office Home 数据集的标签分布统计
    """
    num_clients = len(partition_idxs)
    labels = [data[-1] for data in dataset]

    label_dist = np.zeros((num_clients, num_labels), dtype=int)
    for cid, sids in partition_idxs.items():
        for sid in sids:
            label_dist[cid, labels[sid]] += 1

    print('Label Distribution:')
    print(label_dist)

    if visualize:
        # 原有的散点图
        x = np.tile(np.arange(num_clients), num_labels)
        y = np.repeat(np.arange(num_labels), num_clients)
        size = label_dist[x, y]

        plt.figure(figsize=(12, 8), dpi=dpi)
        plt.scatter(x, y, s=size * resize, color='blue', alpha=0.7)
        plt.xlabel('Clients', fontsize=label_fontsize)
        plt.ylabel('Labels', fontsize=label_fontsize)
        plt.title('Hybrid Shift Label Distribution on Tiny-ImageNet', fontsize=label_fontsize + 2)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'/mnt/sda/PythonProject/LJX_2023/Federate TTA code/'
                    '(Office-home_example)ATP/(ours_Office-home)ATP/exp/'
                    f'Office_home_distribution_png/domain_{environment}.png',
                    dpi=dpi, bbox_inches='tight')
        plt.show()

        # 新增：创建异构性可视化图 - 所有客户端
        filename = f'/mnt/sda/PythonProject/LJX_2023/Federate TTA code/(Office-home_example)ATP/(ours_Office-home)ATP/exp/Office_home_distribution_png/domain_{environment}_heterogeneity_all.png'
        create_heterogeneity_plot_from_partition(
            dataset, num_labels, partition_idxs,
            filename=filename,
            class_id_range=class_id_range,
            client_id_range=client_id_range,
            dpi=dpi,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize
        )


def print_quantity_stat(partition_idxs, visualize=False,
                        dpi=UNIFIED_DPI,
                        fontsize=UNIFIED_LABEL_FONTSIZE-4):
    """
    打印数量统计信息
    """
    quantities = [len(idxs) for idxs in partition_idxs.values()]
    lo = np.min(quantities)
    lo4 = np.quantile(quantities, 1 / 4)
    md = np.median(quantities)
    hi4 = np.quantile(quantities, 3 / 4)
    hi = np.max(quantities)
    mu = np.mean(quantities)
    sd = np.std(quantities, ddof=1)
    print('Quantity Statistics:')
    print(f'  Min: {lo}, Q1: {lo4:.1f}, Median: {md:.1f}, Q3: {hi4:.1f}, Max: {hi}')
    print(f'  Mean ± Std: {mu:.1f} ± {sd:.1f}')

    if visualize:
        plt.figure(figsize=(10, 6), dpi=dpi)
        plt.hist(quantities, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.xlabel('Number of Samples per Client', fontsize=fontsize)
        plt.ylabel('Frequency', fontsize=fontsize)
        plt.title('Client Quantity Distribution', fontsize=fontsize + 2)
        plt.grid(True, alpha=0.3)
        plt.show()


def analyze_dataset_heterogeneity_all_clients(dataset, partition_idxs, num_labels, save_path=True,
                                              class_id_range=UNIFIED_CLASS_ID_RANGE,
                                              client_id_range=UNIFIED_CLIENT_ID_RANGE,
                                              dpi=UNIFIED_DPI,
                                              label_fontsize=UNIFIED_LABEL_FONTSIZE,
                                              tick_fontsize=UNIFIED_TICK_FONTSIZE):
    """
    独立分析所有客户端数据集异构性的函数

    参数:
        class_id_range: Class ID 的显示范围
        client_id_range: Client ID 的显示范围
        dpi: 图片分辨率
        label_fontsize: 坐标轴标签字体大小
        tick_fontsize: 刻度标签字体大小
    """
    print("=== 数据集异构性分析 (所有客户端) ===")

    # 计算基本统计信息
    quantities = [len(idxs) for idxs in partition_idxs.values()]
    lo = np.min(quantities)
    lo4 = np.quantile(quantities, 1 / 4)
    md = np.median(quantities)
    hi4 = np.quantile(quantities, 3 / 4)
    hi = np.max(quantities)
    mu = np.mean(quantities)
    sd = np.std(quantities, ddof=1)
    print('Quantity Statistics:')
    print(f'  Min: {lo}, Q1: {lo4:.1f}, Median: {md:.1f}, Q3: {hi4:.1f}, Max: {hi}')
    print(f'  Mean ± Std: {mu:.1f} ± {sd:.1f}')

    # 创建异构性可视化图 - 所有客户端
    if save_path:
        filename = save_path
    else:
        filename = '/mnt/sda/PythonProject/LYC_Project/FTTA/ATP/picture/dataset_heterogeneity_all_clients.png'

    fig, ax, label_dist = create_heterogeneity_plot_from_partition(
        dataset, num_labels, partition_idxs, filename,
        class_id_range=class_id_range,
        client_id_range=client_id_range,
        dpi=dpi,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize
    )

    # 打印详细统计信息
    print("\n=== 详细统计信息 ===")
    print(f"总客户端数: {len(label_dist)}")
    print(f"总类别数: {label_dist.shape[1]}")
    print(f"总样本数: {label_dist.sum()}")
    print(f"每个客户端平均样本数: {label_dist.sum(axis=1).mean():.1f} ± {label_dist.sum(axis=1).std():.1f}")
    print(f"每个客户端平均类别数: {(label_dist > 0).sum(axis=1).mean():.1f}")
    print(f"数据稀疏度: {(label_dist == 0).sum() / label_dist.size:.3f}")

    # 打印客户端数量分布
    client_sample_counts = label_dist.sum(axis=1)
    print(f"\n客户端样本数量分布:")
    print(f"  最少样本: {client_sample_counts.min()}")
    print(f"  最多样本: {client_sample_counts.max()}")
    print(f"  样本数标准差: {client_sample_counts.std():.1f}")

    return fig, ax, label_dist


def create_large_scale_heterogeneity_plot(dataset, num_labels, partition_idxs, filename=None, clients_per_row=50,
                                          class_id_range=UNIFIED_CLASS_ID_RANGE,
                                          client_id_range=UNIFIED_CLIENT_ID_RANGE,
                                          dpi=UNIFIED_DPI,
                                          label_fontsize=UNIFIED_LABEL_FONTSIZE,
                                          tick_fontsize=UNIFIED_TICK_FONTSIZE):
    """
    专门处理大量客户端的异构性可视化
    """
    return create_heterogeneity_plot_from_partition(
        dataset, num_labels, partition_idxs, filename,
        max_clients_per_row=clients_per_row,
        class_id_range=class_id_range,
        client_id_range=client_id_range,
        dpi=dpi,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize
    )


# 使用示例和测试函数
def test_heterogeneity_plot():
    """
    测试异构性可视化函数
    """
    # 创建测试数据
    num_clients = 50
    num_labels = 65
    num_samples = 1000

    # 创建随机数据集
    dataset = []
    for i in range(num_samples):
        label = np.random.randint(0, num_labels)
        dataset.append((None, label))  # (data, label) 格式

    # 创建随机分区
    partition_idxs = {}
    samples_per_client = num_samples // num_clients
    remaining_samples = num_samples % num_clients

    start_idx = 0
    for client_id in range(num_clients):
        client_samples = samples_per_client
        if client_id < remaining_samples:
            client_samples += 1

        end_idx = start_idx + client_samples
        partition_idxs[client_id] = list(range(start_idx, end_idx))
        start_idx = end_idx

    print("测试数据创建完成")
    print(f"客户端数量: {num_clients}")
    print(f"类别数量: {num_labels}")
    print(f"总样本数: {num_samples}")

    # 测试分析函数 - 使用统一设置的参数
    analyze_dataset_heterogeneity_all_clients(
        dataset=dataset,
        partition_idxs=partition_idxs,
        num_labels=num_labels
    )


# 预设配置函数
def create_publication_quality_plot(dataset, partition_idxs, num_labels, save_path,
                                    plot_type="heterogeneity"):
    """
    创建出版质量的图片

    参数:
        plot_type: "heterogeneity" 或 "distribution"
    """
    if plot_type == "heterogeneity":
        return analyze_dataset_heterogeneity_all_clients(
            dataset=dataset,
            partition_idxs=partition_idxs,
            num_labels=num_labels,
            save_path=save_path
        )
    else:
        return print_label_distribution_stat(
            dataset=dataset,
            num_labels=num_labels,
            partition_idxs=partition_idxs,
            visualize=True
        )


if __name__ == "__main__":
    # 使用示例
    print("数据异构性可视化工具")
    print("=" * 50)
    print(f"统一设置: DPI={UNIFIED_DPI}, 标签字体={UNIFIED_LABEL_FONTSIZE}, 刻度字体={UNIFIED_TICK_FONTSIZE}")
    print(f"显示范围: Class ID 0-{UNIFIED_CLASS_ID_RANGE}, Client ID 0-{UNIFIED_CLIENT_ID_RANGE}")

    # 可以取消注释下面的行来运行测试
    test_heterogeneity_plot()

    # 实际使用示例：
    # analyze_dataset_heterogeneity_all_clients(
    #     dataset=your_dataset,
    #     partition_idxs=your_partition_idxs,
    #     num_labels=your_num_labels,
    #     save_path='./heterogeneity_plot.png'
    # )

    # 或者使用预设配置：
    # create_publication_quality_plot(
    #     dataset=your_dataset,
    #     partition_idxs=your_partition_idxs,
    #     num_labels=your_num_labels,
    #     save_path='./publication_quality_plot.png',
    #     plot_type="heterogeneity"
    # )

    print("请根据您的实际数据调用相应的函数")