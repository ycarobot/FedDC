"""
Step Partition
"""

import numpy as np
from scipy.stats import dirichlet

from .utils import get_labels

import itertools


def step_partition(dataset, num_labels, num_clients, num_major, alpha):
    """
    :param dataset: Dataset
    :param num_clients: number of clients ()
    :param alpha: concentration score. Larger alpha -> more IID
    :return:
    """

    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    # label skewness: control each client's label distribution (separately)
    prior = num_samples_per_label / num_samples_per_label.sum()

    if alpha == float('inf'):
        matrix = np.zeros((num_clients, num_labels))
        alpha = 2
    else:
        matrix = np.ones((num_clients, num_labels))

    print(num_major)

    if num_clients == 300 and num_labels == 10 and num_major == 2:  # CIFAR-10、CIBNIC-10 experiments
        for cid, label_ids in enumerate(itertools.product(range(num_labels), repeat=num_major)):
            for label_id in label_ids:
                matrix[cid * 3:(cid + 1) * 3, label_id] += (alpha - 1)

    elif num_clients == 10 and num_labels == 10 and num_major == 2:  # Digits experiments
        for cid in range(10):
            matrix[cid, cid] += (alpha - 1)
            matrix[cid, (cid + 1) % 10] += (alpha - 1)

    elif num_clients == 7 and num_labels == 7 and num_major == 2:  # PACS experiments
        for cid in range(7):
            matrix[cid, cid] += (alpha - 1)
            matrix[cid, (cid + 1) % 7] += (alpha - 1)
    elif num_clients == 7 and num_labels == 31 and num_major == 2:  # Offcie31 experiments
        for cid in range(7):
            major_labels = [(cid * num_major + i) % num_labels for i in range(num_major)]
            for label in major_labels:
                matrix[cid, label] += (alpha -1)
            next_label = (cid * num_major + num_major) % num_labels
            matrix[cid, next_label] += (alpha - 1)

    elif num_clients == 7 and num_labels == 65 and num_major == 2:  # Offcie_Home experiments
        for cid in range(7):
            major_labels = [(cid * num_major + i) % num_labels for i in range(num_major)]
            for label in major_labels:
                matrix[cid, label] += (alpha -1)
            next_label = (cid * num_major + num_major) % num_labels
            matrix[cid, next_label] += (alpha - 1)

    elif num_clients == 7 and num_labels == 10 and num_major == 2:  # Offcie_caltech10 experiments
        for cid in range(7):
            major_labels = [(cid * num_major + i) % num_labels for i in range(num_major)]
            for label in major_labels:
                matrix[cid, label] += (alpha - 1)
            next_label = (cid * num_major + num_major) % num_labels
            matrix[cid, next_label] += (alpha - 1)

    elif num_clients == 300 and num_labels == 100:  # CIFAR-100 experiments
        for i in range(3):
            label_gap = 1 + 2 * i
            client_init = i * 100
            for label_init in range(100):
                cid = client_init + label_init
                for j in range(num_major):
                    label_id = (label_init + label_gap * j) % 100
                    matrix[cid, label_id] += (alpha - 1)
        # dirichlet_alpha = 0.1
        # matrix = dirichlet.rvs([dirichlet_alpha] * num_labels, size=num_clients)
        # matrix = matrix * num_clients

    elif num_clients == 200 and num_labels == 200:  # TinyImageNet-200 experiments
        for i in range(2):  # 将300个客户端分成3组
            label_gap = 1 + 2 * i  # 计算标签间隔（1,3,5）
            client_init = i * 100  # 每组100个客户端

            for label_init in range(100):  # 每组处理100个标签（共200个标签）
                cid = client_init + label_init  # 客户端ID

                for j in range(num_major):  # 为每个客户端分配num_major个主要标签
                    # 计算标签ID，确保在0-199范围内
                    label_id = (label_init * 2 + label_gap * j) % 200
                    matrix[cid, label_id] += (alpha - 1)
        # dirichlet_alpha = 0.1
        # matrix = dirichlet.rvs([dirichlet_alpha] * num_labels, size=num_clients)
        # matrix = matrix * num_clients



    elif num_clients == 42 and num_labels == 345 and num_major == 2:  # DomainNet experiments

        for cid in range(7):
            major_labels = [(cid * num_major + i) % num_labels for i in range(num_major)]
            for label in major_labels:
                matrix[cid, label] += (alpha - 1)
            next_label = (cid * num_major + num_major) % num_labels
            matrix[cid, next_label] += (alpha - 1)



    # elif num_clients == 300 and num_labels == 1000:  # ImageNet experiments
    #     for i in range(3):  # 可以根据实际情况调整分组数量
    #         label_gap = 1 + 2 * i
    #         client_init = i * 100
    #         for label_init in range(100):
    #             cid = client_init + label_init
    #             for j in range(num_major):
    #                 label_id = (label_init + label_gap * j) % 1000
    #                 matrix[cid, label_id] += (alpha - 1)
    elif num_clients == 300 and num_labels == 1000:  # ImageNet experiments
        # 更合理的标签分配策略
        clients_per_group = num_clients // 10  # 每组30个客户端

        for group in range(10):  # 将1000个类别分为10组
            start_label = group * 100
            end_label = (group + 1) * 100

            for cid_in_group in range(clients_per_group):
                cid = group * clients_per_group + cid_in_group

                # 为每个客户端分配主要标签
                for j in range(num_major):
                    # 确保标签在当前组内均匀分布
                    label_id = start_label + (cid_in_group + j * clients_per_group) % 100
                    matrix[cid, label_id] += (alpha - 1)

    elif num_clients == 200 and num_labels == 101:  # Food-101 experiments (优化版)
        # 使用更均匀的标签分布策略
        labels_per_client = num_major  # 每个客户端的主要标签数

        # 将101个标签循环分配给200个客户端
        for cid in range(num_clients):
            # 基于客户端ID计算起始标签
            base_label = (cid * labels_per_client) % num_labels

            for j in range(num_major):
                # 分配主要标签
                label_id = (base_label + j) % num_labels
                matrix[cid, label_id] += (alpha - 1)

            # 为了增加多样性，可以再添加一个相邻标签
            extra_label = (base_label + num_major) % num_labels
            matrix[cid, extra_label] += (alpha - 1)

    elif num_clients == 28 and num_labels == 196 and num_major == 2:
        """
        Stanford Cars数据集划分策略：
        - 196个类别分配给28个客户端
        - 每个客户端分配2个主要类别
        - 使用循环分配确保类别均匀分布
        """
        # 计算每个客户端应该分配的类别跨度
        labels_per_client = (num_labels * num_major) // num_clients  # 14个类别/客户端

        for cid in range(num_clients):
            # 基于客户端ID计算主要类别的起始位置
            base_label = (cid * labels_per_client) % num_labels

            # 为每个客户端分配num_major个主要类别
            for j in range(num_major):
                label_id = (base_label + j * (labels_per_client // num_major)) % num_labels
                matrix[cid, label_id] += (alpha - 1)


    else:
        raise NotImplementedError

    # normalizing matrix
    matrix = matrix / matrix.sum(axis=0)

    # cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.5).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for cid in range(num_clients):
        idxs = []
        for label in range(num_labels):
            idxs.append(idxs_by_label[label][cumulate[cid, label]:cumulate[cid + 1, label]])

        partition_idxs[cid] = np.concatenate(idxs)

    return partition_idxs
