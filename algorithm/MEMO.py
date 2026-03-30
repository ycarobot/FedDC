# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# from copy import deepcopy
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import numpy as np
#
# from model import create_model, create_loss, create_metric, create_optimizer
# from utils.third_party import aug_digit, aug_cifar, aug_pacs, aug_tiny_imagenet
#
# from .Base import BaseServer, BaseClient
# from .TTABase import TTABaseServer
#
#
# class MEMOServer(TTABaseServer):
#
#     def __init__(self, train_datasets, test_datasets, args):
#         TTABaseServer.__init__(self, train_datasets, test_datasets, args)
#
#         self.train_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
#         self.test_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
#
#         # load a pre-trained model (loading in main.py)
#         self.model = create_model(args)
#
#         prior = args.prior_strength / (args.prior_strength + 1)
#
#         # self.model.change_bn(mode='prior', prior=prior)
#
#         self.model.eval()
#
#
# class MEMOClient(BaseClient):
#
#     def __init__(self, cid, datasets, args):
#         BaseClient.__init__(self, cid, datasets, args)
#
#         if args.dataset == 'cifar10':
#             self.tr_pre = transforms.Compose([
#                 transforms.Normalize((0, 0, 0), (1 / 0.2470, 1 / 0.2435, 1 / 0.2616)),
#                 transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
#                 transforms.ToPILImage()
#             ])
#             self.tr_post = transforms.Compose([
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
#             ])
#             self.aug = aug_cifar
#
#         elif args.dataset == 'digit':
#             self.tr_pre = transforms.Compose([
#                 transforms.Normalize((0, 0, 0), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
#                 transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
#                 transforms.ToPILImage()
#             ])
#             self.tr_post = transforms.Compose([
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # mean and std of each channel
#             ])
#             self.aug = aug_digit
#
#         elif args.dataset == 'cifar100':
#             self.tr_pre = transforms.Compose([
#                 transforms.Normalize((0, 0, 0), (1 / 0.2673, 1 / 0.2546, 1 / 0.2762)),
#                 transforms.Normalize((-0.5071, -0.4866, -0.4409), (1, 1, 1)),
#                 transforms.ToPILImage()
#             ])
#             self.tr_post = transforms.Compose([
#                 transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # mean and std of each channel
#             ])
#             self.aug = aug_cifar
#
#         elif args.dataset == 'pacs_aug':
#             self.tr_pre = transforms.Compose([
#                 transforms.Normalize((0, 0, 0), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
#                 transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1)),
#                 transforms.ToPILImage()
#             ])
#             self.tr_post = transforms.Compose([
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std of each channel
#             ])
#             self.aug = aug_pacs
#
#
#         elif args.dataset == 'office_home_aug':
#
#             self.tr_pre = transforms.Compose([
#
#                 transforms.Normalize((0, 0, 0), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
#
#                 transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1)),
#
#                 transforms.ToPILImage()
#
#             ])
#
#             self.tr_post = transforms.Compose([
#
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 使用与PACS相同的均值和标准差
#
#             ])
#
#             self.aug = aug_pacs
#
#         elif args.dataset == 'tiny_imagenet':
#             # Tiny-ImageNet的均值和标准差（通常与ImageNet一致）
#             mean = (0.485, 0.456, 0.406)
#             std = (0.229, 0.224, 0.225)
#
#             # tr_pre：将标准化的图像还原为原始像素空间（0-255），以便进行数据增强
#             self.tr_pre = transforms.Compose([
#                 transforms.Normalize((0, 0, 0), (1 / std[0], 1 / std[1], 1 / std[2])),  # 逆标准化（去除标准差缩放）
#                 transforms.Normalize((-mean[0], -mean[1], -mean[2]), (1, 1, 1)),  # 逆标准化（去除均值偏移）
#                 transforms.ToPILImage()  # 转换为PIL图像，方便后续增强操作
#             ])
#
#             # tr_post：对增强后的图像重新标准化，符合模型输入要求
#             self.tr_post = transforms.Compose([
#                 transforms.Normalize(mean, std),  # 应用ImageNet的均值和标准差
#             ])
#
#             # 数据增强函数（需确保utils.third_party中存在aug_tiny_imagenet）
#             self.aug = aug_tiny_imagenet
#
#         else:
#             raise NotImplementedError('MEMO requires data in its original space')
#         self.batch_size = 1
#
#         self.dataloaders = {}
#         for key, dataset in self.datasets.items():
#             if key in ['train', ]:
#                 # for training set, we shuffle the data, we drop too small batch in the training if necessary
#                 self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
#                                                    num_workers=self.num_workers)
#
#             elif key in ['valid', 'test', ]:
#                 # for testing set, it is not necessary to shuffle
#                 self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
#                                                    num_workers=self.num_workers)
#
#
#     def adapt_single(self, model, image, optimizer, args):
#         model.eval()
#         image = self.tr_pre(image)
#         inputs = [self.tr_post(self.aug(image)) for _ in range(args.memo_aug_size)]
#         inputs = torch.stack(inputs).to(self.device)
#
#         # print(inputs.shape)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#
#         loss, logits = marginal_entropy(outputs)
#
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#
#     def local_eval(self, model, args, dataset='test'):
#
#         spv_loss_func = create_loss('ce')
#         metric_func = create_metric('acc')
#         optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)
#
#         total_examples, total_loss, total_metric = 0, 0, 0
#
#         dataloader = self.dataloaders[dataset]
#
#         state = deepcopy(model.state_dict())
#
#         for *X, Y in dataloader:
#             model.load_state_dict(state)
#
#             image = X[0][0]  # get the first sample (only sample)
#             Y = Y.to(self.device)
#
#             self.adapt_single(model, image, optimizer, args)
#
#             model.eval()
#
#             X = [x.to(self.device) for x in X]
#
#             # print(X[0].shape)
#
#             with torch.no_grad():
#                 logits = model(*X)
#                 spv_loss = spv_loss_func(logits, Y)
#                 metric = metric_func(logits, Y)
#                 num_examples = len(X[0])
#
#                 total_examples += num_examples
#                 total_loss += spv_loss.item() * num_examples
#                 total_metric += metric.item() * num_examples
#
#         avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
#
#         # print(avg_metric)
#
#         return avg_loss, avg_metric, total_examples
#
#
#
#
#
# def marginal_entropy(outputs):
#     logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
#     avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
#     min_real = torch.finfo(avg_logits.dtype).min
#     avg_logits = torch.clamp(avg_logits, min=min_real)
#     return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import warnings

# 忽略requests版本警告
warnings.filterwarnings('ignore', category=UserWarning, module='requests')

from model import create_model, create_loss, create_metric, create_optimizer
from utils.third_party import aug_digit, aug_cifar, aug_pacs, aug_tiny_imagenet

# 为新数据集添加默认增强函数（如果utils中没有，这里提供基础实现）
try:
    from utils.third_party import aug_food101, aug_stanfordcars
except ImportError:
    # 基础数据增强实现（兼容ImageNet系列数据集）
    def aug_food101(img):
        """Food101数据增强（仿ImageNet）"""
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
        return transform(img)


    def aug_stanfordcars(img):
        """StanfordCars数据增强（仿ImageNet）"""
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
        return transform(img)

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class MEMOServer(TTABaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        # 打印当前数据集配置，方便调试
        print(f"[MEMOServer] Initializing for dataset: {args.dataset}")

        self.train_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)

        prior = args.prior_strength / (args.prior_strength + 1)
        # self.model.change_bn(mode='prior', prior=prior)
        self.model.eval()


class MEMOClient(BaseClient):
    def __init__(self, cid, datasets, args):
        BaseClient.__init__(self, cid, datasets, args)

        # 数据集均值和标准差配置（统一使用ImageNet标准，适配大多数预训练模型）
        dataset_configs = {
            # 原有数据集
            'cifar10': {
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2470, 0.2435, 0.2616),
                'aug': aug_cifar
            },
            'digit': {
                'mean': (0.5, 0.5, 0.5),
                'std': (0.5, 0.5, 0.5),
                'aug': aug_digit
            },
            'cifar100': {
                'mean': (0.5071, 0.4866, 0.4409),
                'std': (0.2673, 0.2564, 0.2762),
                'aug': aug_cifar
            },
            'pacs_aug': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'aug': aug_pacs
            },
            'office_home_aug': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'aug': aug_pacs
            },
            'tiny_imagenet': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'aug': aug_tiny_imagenet
            },
            # 新增数据集
            'food101': {
                'mean': (0.485, 0.456, 0.406),  # ImageNet均值
                'std': (0.229, 0.224, 0.225),  # ImageNet标准差
                'aug': aug_food101
            },
            'stanfordcars': {
                'mean': (0.485, 0.456, 0.406),  # ImageNet均值
                'std': (0.229, 0.224, 0.225),  # ImageNet标准差
                'aug': aug_stanfordcars
            }
        }

        # 获取当前数据集配置，默认使用ImageNet配置
        if args.dataset in dataset_configs:
            config = dataset_configs[args.dataset]
            mean = config['mean']
            std = config['std']
            self.aug = config['aug']
            #print(f"[MEMOClient-{cid}] Using {args.dataset} config: mean={mean}, std={std}")
        else:
            # 未知数据集使用ImageNet默认配置，不再抛异常
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.aug = aug_food101  # 通用增强函数
            warnings.warn(f"[MEMOClient-{cid}] Dataset {args.dataset} not found, using ImageNet default config")

        # 构建逆标准化+标准化转换管道
        self.tr_pre = transforms.Compose([
            # 逆标准化：还原到原始像素空间（0-255）
            transforms.Normalize((0, 0, 0), (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.Normalize((-mean[0], -mean[1], -mean[2]), (1, 1, 1)),
            transforms.ToPILImage()
        ])

        self.tr_post = transforms.Compose([
            # 重新标准化，符合模型输入要求
            transforms.Normalize(mean, std),
        ])

        self.batch_size = 1

        # 初始化数据加载器
        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            shuffle = key == 'train'
            self.dataloaders[key] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=False,
                num_workers=self.num_workers
            )

    def adapt_single(self, model, image, optimizer, args):
        """单样本自适应（核心MEMO逻辑）"""
        model.eval()

        # 确保image是CPU张量（ToPILImage不支持CUDA）
        if image.is_cuda:
            image = image.cpu()

        # 逆标准化→增强→重新标准化
        image = self.tr_pre(image)
        inputs = [self.tr_post(self.aug(image)) for _ in range(args.memo_aug_size)]
        inputs = torch.stack(inputs).to(self.device)

        # 边际熵损失计算与反向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, _ = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def local_eval(self, model, args, dataset='test'):
        """本地评估主逻辑"""
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        total_examples, total_loss, total_metric = 0, 0, 0
        dataloader = self.dataloaders[dataset]
        state = deepcopy(model.state_dict())

        for *X, Y in dataloader:
            # 重置模型状态
            model.load_state_dict(state)

            # 单样本自适应
            image = X[0][0]  # 获取单个样本
            Y = Y.to(self.device)
            self.adapt_single(model, image, optimizer, args)

            # 评估
            model.eval()
            X = [x.to(self.device) for x in X]

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)

                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

        # 计算平均损失和准确率
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        avg_metric = total_metric / total_examples if total_examples > 0 else 0

        return avg_loss, avg_metric, total_examples


def marginal_entropy(outputs):
    """计算边际熵（MEMO核心损失函数）"""
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits