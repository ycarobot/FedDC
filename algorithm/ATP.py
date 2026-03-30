# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from tqdm import tqdm
# # from copy import deepcopy
# #
# # from model import create_model, create_loss, create_metric, create_optimizer
# # from model.MyBatchNorm2d import MyBatchNorm2d
# #
# # from .Base import BaseServer, BaseClient
# # from .TTPBase import TTPBaseServer
# #
# #
# # class ATPServer(TTPBaseServer):
# #     """
# #     A class for debugging
# #     """
# #
# #     def __init__(self, train_datasets, test_datasets, args):
# #         TTPBaseServer.__init__(self, train_datasets, test_datasets, args)
# #
# #         self.train_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
# #         self.test_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
# #
# #         # load a pre-trained model
# #         self.model = create_model(args)
# #
# #         self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
# #         # which has identical behavior, but support taking gradient
# #         self.model.eval()
# #
# #         num_ar = len([name for name, params in self.model.named_parameters() if params.requires_grad])
# #         print('Dimension of Adapt Rate:', num_ar)
# #
# #         args.idx_params = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' not in name]
# #         args.idx_stats = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' in name]
# #
# #         print('  - Params:', len(args.idx_params))
# #         print('  - Stats:', len(args.idx_stats))
# #
# #         if args.verbose:
# #             print([name for name, params in self.model.named_parameters() if params.requires_grad])
# #
# #         self.adapt_lrs = torch.zeros(len(self.model.trainable_parameters())).to(args.device)
# #
# #
# # class ATPClient(BaseClient):
# #     """
# #     A class for debug
# #     """
# #
# #     def __init__(self, cid, datasets, args):
# #         BaseClient.__init__(self, cid, datasets, args)
# #
# #         self.lr = args.lm_lr  # the learning rate of adaptation rates
# #
# #     def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
# #
# #         model.eval()
# #
# #         logits = model(*X)
# #
# #         loss = unspv_loss_func(logits, Y)
# #
# #         loss.backward()
# #
# #         model.set_running_stat_grads()
# #
# #         unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]
# #
# #         with torch.no_grad():
# #             for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
# #                 p -= adapt_lrs[i] * g
# #
# #         model.zero_grad()
# #
# #         model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.
# #
# #         return unspv_grad
# #
# #     def local_train(self, model, adapt_lrs, args, dataset='test'):
# #
# #         unspv_loss_func = create_loss('ent')
# #         spv_loss_func = create_loss('ce')
# #         metric_func = create_metric('acc')
# #
# #         total_examples, total_loss, total_metric = 0, 0, 0
# #
# #         dataloader = self.dataloaders[dataset]
# #         num_data = self.num_data[dataset]
# #
# #         state = deepcopy(model.state_dict())
# #
# #         for *X, Y in dataloader:
# #             model.load_state_dict(state)
# #
# #             # Get a batch of data
# #             X = [x.to(self.device) for x in X]
# #             Y = Y.to(self.device)
# #
# #             # 1. unsupervised adaptation
# #
# #             unspv_grad = self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)
# #
# #             # 2. supervised evaluation
# #
# #             model.eval()
# #
# #             logits = model(*X)
# #             spv_loss = spv_loss_func(logits, Y)
# #
# #             spv_grad = torch.autograd.grad(spv_loss, model.trainable_parameters())
# #
# #             # 3. update the adaptation rate
# #             with torch.no_grad():
# #
# #                 # manual resize
# #
# #                 if args.grad_norm == 'none':
# #                     g = torch.zeros_like(adapt_lrs)
# #                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
# #                         g[i] += (g1 * g2).sum()
# #
# #                 elif args.grad_norm == 'numel':
# #                     g = torch.zeros_like(adapt_lrs)
# #                     l = torch.zeros_like(adapt_lrs)
# #                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
# #                         g[i] += (g1 * g2).sum()
# #                         l[i] += g1.numel()
# #
# #                     g /= l
# #
# #                 elif args.grad_norm == 'sqrt_numel':
# #                     g = torch.zeros_like(adapt_lrs)
# #                     l = torch.zeros_like(adapt_lrs)
# #                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
# #                         g[i] += (g1 * g2).sum()
# #                         l[i] += g1.numel()
# #
# #                     g /= torch.sqrt(l)
# #
# #                 elif args.grad_norm == 'manual':
# #                     g = torch.zeros_like(adapt_lrs)
# #                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
# #                         if i in args.idx_params:
# #                             g[i] += (g1 * g2).sum()
# #                         elif i in args.idx_stats:
# #                             g[i] += 100 * (g1 * g2).sum()
# #
# #                 elif args.grad_norm == 'params_only':
# #                     g = torch.zeros_like(adapt_lrs)
# #                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
# #                         if i in args.idx_params:
# #                             g[i] += (g1 * g2).sum()
# #                             g[i] /= g1.numel()
# #
# #                 elif args.grad_norm == 'stats_only':
# #                     g = torch.zeros_like(adapt_lrs)
# #                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
# #                         if i in args.idx_stats:
# #                             g[i] += (g1 * g2).sum()
# #                             g[i] /= g1.numel()
# #
# #                 else:
# #                     raise NotImplementedError
# #
# #                 model_type = args.model
# #
# #                 # 1. ResNet 系列（--model resnet18 / resnet50 等）：用 +=
# #                 if 'resnet' in model_type:
# #                     adapt_lrs += self.lr * g
# #                     # 可选：打印调试信息，验证判断逻辑
# #                     # print(f"Model: {args.model} (ResNet), adapt_lrs updated with '+='")
# #
# #                 # 2. ViT 系列（--model vit_b_16 / vit_l_32 等）：用 -=
# #                 elif 'vit' in model_type:
# #                     adapt_lrs += self.lr * g
# #
# #             with torch.no_grad():
# #                 num_examples = len(X[0])
# #                 total_examples += num_examples
# #                 total_loss += spv_loss.item() * num_examples
# #                 metric = metric_func(logits, Y)
# #                 total_metric += metric.item() * num_examples
# #
# #         avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
# #
# #         return avg_loss, avg_metric, num_data
# #
# #     def local_eval(self, model, adapt_lrs, args, dataset='test'):
# #
# #         unspv_loss_func = create_loss('ent')
# #         spv_loss_func = create_loss('ce')
# #         metric_func = create_metric('acc')
# #
# #         total_examples, total_loss, total_metric = 0, 0, 0
# #
# #         dataloader = self.dataloaders[dataset]
# #         num_data = self.num_data[dataset]
# #
# #         state = deepcopy(model.state_dict())
# #
# #         for i, (*X, Y) in enumerate(dataloader):
# #             model.load_state_dict(state)
# #
# #             # Get a batch of data
# #             X = [x.to(self.device) for x in X]
# #             Y = Y.to(self.device)
# #
# #             # 1. unsupervised adaptation
# #
# #             self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)
# #
# #             # 2. supervised evaluation
# #
# #             model.eval()
# #
# #             with torch.no_grad():
# #                 logits = model(*X)
# #                 spv_loss = spv_loss_func(logits, Y)
# #
# #                 # record the loss and accuracy
# #                 num_examples = len(X[0])
# #                 total_examples += num_examples
# #                 total_loss += spv_loss.item() * num_examples
# #                 metric = metric_func(logits, Y)
# #                 total_metric += metric.item() * num_examples
# #
# #         avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
# #
# #         return avg_loss, avg_metric, num_data
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# from copy import deepcopy
#
# # 框架依赖导入
# from model import create_model, create_loss, create_metric, create_optimizer
# from model.MyBatchNorm2d import MyBatchNorm2d
#
# from .Base import BaseServer, BaseClient
# from .TTPBase import TTPBaseServer
#
#
# class ATPServer(TTPBaseServer):
#     """
#     A class for debugging
#     """
#
#     def __init__(self, train_datasets, test_datasets, args):
#         TTPBaseServer.__init__(self, train_datasets, test_datasets, args)
#
#         self.train_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
#         self.test_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
#
#         # load a pre-trained model
#         self.model = create_model(args)
#
#         # 修复：兼容模型可能没有change_bn方法的情况
#         if hasattr(self.model, 'change_bn'):
#             self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
#             # which has identical behavior, but support taking gradient
#         self.model.eval()
#
#         # 统计可训练参数数量
#         num_ar = len([name for name, params in self.model.named_parameters() if params.requires_grad])
#         print('Dimension of Adapt Rate:', num_ar)
#
#         # 修复：参数索引筛选逻辑
#         args.idx_params = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' not in name]
#         args.idx_stats = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' in name]
#
#         print('  - Params:', len(args.idx_params))
#         print('  - Stats:', len(args.idx_stats))
#
#         if args.verbose:
#             print([name for name, params in self.model.named_parameters() if params.requires_grad])
#
#         # ========== 核心修复 ==========
#         # 错误原因：trainable_parameters是列表而非函数，不能加括号调用
#         # 方案1：如果模型有trainable_parameters列表属性
#         if hasattr(self.model, 'trainable_parameters') and isinstance(self.model.trainable_parameters, list):
#             self.adapt_lrs = torch.zeros(len(self.model.trainable_parameters)).to(args.device)
#         # 方案2：兼容情况，直接从named_parameters获取可训练参数
#         else:
#             trainable_params = [p for p in self.model.parameters() if p.requires_grad]
#             self.adapt_lrs = torch.zeros(len(trainable_params)).to(args.device)
#
#
# class ATPClient(BaseClient):
#     """
#     A class for debug
#     """
#
#     def __init__(self, cid, datasets, args):
#         BaseClient.__init__(self, cid, datasets, args)
#
#         self.lr = args.lm_lr  # the learning rate of adaptation rates
#         # 修复：确保device属性存在
#         self.device = args.device if hasattr(args, 'device') else torch.device(
#             'cuda' if torch.cuda.is_available() else 'cpu')
#
#     def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
#         """单步适配（修复潜在的BN相关问题）"""
#         model.eval()
#
#         logits = model(*X)
#         loss = unspv_loss_func(logits, Y)
#         loss.backward()
#
#         # 修复：兼容模型可能没有set_running_stat_grads方法的情况
#         if hasattr(model, 'set_running_stat_grads'):
#             model.set_running_stat_grads()
#
#         # 获取梯度（兼容不同模型参数获取方式）
#         if hasattr(model, 'trainable_parameters') and isinstance(model.trainable_parameters, list):
#             unspv_grad = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
#                           for p in model.trainable_parameters]
#         else:
#             unspv_grad = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
#                           for p in model.parameters() if p.requires_grad]
#
#         # 适配率更新
#         with torch.no_grad():
#             # 确保梯度和适配率长度匹配
#             assert len(unspv_grad) == len(adapt_lrs), f"梯度长度{len(unspv_grad)}与适配率长度{len(adapt_lrs)}不匹配"
#
#             for i, (p, g) in enumerate(zip(model.parameters() if not hasattr(model,
#                                                                              'trainable_parameters') else model.trainable_parameters,
#                                            unspv_grad)):
#                 if p.requires_grad:
#                     p -= adapt_lrs[i] * g
#
#         model.zero_grad()
#
#         # 修复：兼容模型可能没有clip_bn_running_vars方法的情况
#         if hasattr(model, 'clip_bn_running_vars'):
#             model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.
#
#         return unspv_grad
#
#     def local_train(self, model, adapt_lrs, args, dataset='test'):
#         """本地训练（增加鲁棒性处理）"""
#         unspv_loss_func = create_loss('ent')
#         spv_loss_func = create_loss('ce')
#         metric_func = create_metric('acc')
#
#         total_examples, total_loss, total_metric = 0, 0, 0
#
#         dataloader = self.dataloaders[dataset]
#         num_data = self.num_data[dataset] if hasattr(self, 'num_data') and dataset in self.num_data else len(
#             dataloader.dataset)
#
#         state = deepcopy(model.state_dict())
#
#         for *X, Y in dataloader:
#             model.load_state_dict(state)
#
#             # Get a batch of data
#             X = [x.to(self.device) for x in X]
#             Y = Y.to(self.device)
#
#             # 1. unsupervised adaptation
#             unspv_grad = self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)
#
#             # 2. supervised evaluation
#             model.eval()
#
#             logits = model(*X)
#             spv_loss = spv_loss_func(logits, Y)
#
#             # 获取监督梯度（增加梯度存在性检查）
#             try:
#                 if hasattr(model, 'trainable_parameters') and isinstance(model.trainable_parameters, list):
#                     spv_grad = torch.autograd.grad(spv_loss, model.trainable_parameters)
#                 else:
#                     spv_grad = torch.autograd.grad(spv_loss, [p for p in model.parameters() if p.requires_grad])
#             except RuntimeError:
#                 # 梯度计算失败时使用0梯度
#                 spv_grad = [torch.zeros_like(g) for g in unspv_grad]
#
#             # 3. update the adaptation rate
#             with torch.no_grad():
#                 # 确保梯度长度匹配
#                 if len(spv_grad) != len(unspv_grad):
#                     print(f"警告：监督梯度长度{len(spv_grad)}与无监督梯度长度{len(unspv_grad)}不匹配，使用0梯度")
#                     spv_grad = [torch.zeros_like(g) for g in unspv_grad]
#
#                 # manual resize
#                 g = torch.zeros_like(adapt_lrs)
#
#                 if args.grad_norm == 'none':
#                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
#                         g[i] += (g1 * g2).sum()
#
#                 elif args.grad_norm == 'numel':
#                     l = torch.zeros_like(adapt_lrs)
#                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
#                         g[i] += (g1 * g2).sum()
#                         l[i] += g1.numel()
#                     g /= l.clamp(min=1)  # 防止除零
#
#                 elif args.grad_norm == 'sqrt_numel':
#                     l = torch.zeros_like(adapt_lrs)
#                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
#                         g[i] += (g1 * g2).sum()
#                         l[i] += g1.numel()
#                     g /= torch.sqrt(l.clamp(min=1))  # 防止除零和开方负数
#
#                 elif args.grad_norm == 'manual':
#                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
#                         if i in args.idx_params:
#                             g[i] += (g1 * g2).sum()
#                         elif i in args.idx_stats:
#                             g[i] += 100 * (g1 * g2).sum()
#
#                 elif args.grad_norm == 'params_only':
#                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
#                         if i in args.idx_params:
#                             g[i] += (g1 * g2).sum()
#                             g[i] /= max(g1.numel(), 1)  # 防止除零
#
#                 elif args.grad_norm == 'stats_only':
#                     for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
#                         if i in args.idx_stats:
#                             g[i] += (g1 * g2).sum()
#                             g[i] /= max(g1.numel(), 1)  # 防止除零
#
#                 else:
#                     raise NotImplementedError(f"不支持的梯度归一化方式：{args.grad_norm}")
#
#                 model_type = args.model
#
#                 # 1. ResNet 系列（--model resnet18 / resnet50 等）：用 +=
#                 if 'resnet' in model_type:
#                     adapt_lrs += self.lr * g
#
#                 # 2. ViT 系列（--model vit_b_16 / vit_l_32 等）：用 +=
#                 elif 'vit' in model_type:
#                     adapt_lrs += self.lr * g
#
#             # 记录损失和准确率
#             with torch.no_grad():
#                 num_examples = len(X[0])
#                 total_examples += num_examples
#                 total_loss += spv_loss.item() * num_examples
#                 metric = metric_func(logits, Y)
#                 total_metric += metric.item() * num_examples
#
#         # 防止除零
#         avg_loss = total_loss / max(total_examples, 1)
#         avg_metric = total_metric / max(total_examples, 1)
#
#         return avg_loss, avg_metric, num_data
#
#     def local_eval(self, model, adapt_lrs, args, dataset='test'):
#         """本地评估（增加鲁棒性处理）"""
#         unspv_loss_func = create_loss('ent')
#         spv_loss_func = create_loss('ce')
#         metric_func = create_metric('acc')
#
#         total_examples, total_loss, total_metric = 0, 0, 0
#
#         dataloader = self.dataloaders[dataset]
#         num_data = self.num_data[dataset] if hasattr(self, 'num_data') and dataset in self.num_data else len(
#             dataloader.dataset)
#
#         state = deepcopy(model.state_dict())
#
#         for i, (*X, Y) in enumerate(dataloader):
#             model.load_state_dict(state)
#
#             # Get a batch of data
#             X = [x.to(self.device) for x in X]
#             Y = Y.to(self.device)
#
#             # 1. unsupervised adaptation
#             self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)
#
#             # 2. supervised evaluation
#             model.eval()
#
#             with torch.no_grad():
#                 logits = model(*X)
#                 spv_loss = spv_loss_func(logits, Y)
#
#                 # record the loss and accuracy
#                 num_examples = len(X[0])
#                 total_examples += num_examples
#                 total_loss += spv_loss.item() * num_examples
#                 metric = metric_func(logits, Y)
#                 total_metric += metric.item() * num_examples
#
#         # 防止除零
#         avg_loss = total_loss / max(total_examples, 1)
#         avg_metric = total_metric / max(total_examples, 1)
#
#         return avg_loss, avg_metric, num_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

# 框架依赖导入
from model import create_model, create_loss, create_metric, create_optimizer
from model.MyBatchNorm2d import MyBatchNorm2d

from .Base import BaseServer, BaseClient
from .TTPBase import TTPBaseServer


class ATPServer(TTPBaseServer):
    """
    A class for debugging
    """

    def __init__(self, train_datasets, test_datasets, args):
        TTPBaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        # 修复：兼容模型可能没有change_bn方法的情况
        if hasattr(self.model, 'change_bn'):
            self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm
        self.model.eval()

        # 统计可训练参数数量
        num_ar = len([name for name, params in self.model.named_parameters() if params.requires_grad])
        print('Dimension of Adapt Rate:', num_ar)

        # 修复：参数索引筛选逻辑
        args.idx_params = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' not in name]
        args.idx_stats = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' in name]

        print('  - Params:', len(args.idx_params))
        print('  - Stats:', len(args.idx_stats))

        if args.verbose:
            print([name for name, params in self.model.named_parameters() if params.requires_grad])

        # ========== 核心修复：正确处理trainable_parameters ==========
        # 方案：统一获取可训练参数列表（优先调用方法，再降级到默认方式）
        def get_trainable_params(model):
            """安全获取模型可训练参数列表"""
            try:
                # 情况1：trainable_parameters是方法（需要调用）
                if hasattr(model, 'trainable_parameters') and callable(model.trainable_parameters):
                    return model.trainable_parameters()
                # 情况2：trainable_parameters是属性
                elif hasattr(model, 'trainable_parameters'):
                    return model.trainable_parameters
                # 情况3：默认方式获取
                else:
                    return [p for p in model.parameters() if p.requires_grad]
            except:
                # 异常时降级
                return [p for p in model.parameters() if p.requires_grad]

        self.trainable_params = get_trainable_params(self.model)
        self.adapt_lrs = torch.zeros(len(self.trainable_params)).to(args.device)


class ATPClient(BaseClient):
    """
    A class for debug
    """

    def __init__(self, cid, datasets, args):
        BaseClient.__init__(self, cid, datasets, args)

        self.lr = args.lm_lr  # the learning rate of adaptation rates
        # 修复：确保device属性存在
        self.device = args.device if hasattr(args, 'device') else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def get_trainable_params(self, model):
        """工具方法：安全获取模型可训练参数列表（和Server端保持一致）"""
        try:
            if hasattr(model, 'trainable_parameters') and callable(model.trainable_parameters):
                return model.trainable_parameters()
            elif hasattr(model, 'trainable_parameters'):
                return model.trainable_parameters
            else:
                return [p for p in model.parameters() if p.requires_grad]
        except:
            return [p for p in model.parameters() if p.requires_grad]

    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
        """单步适配（彻底修复迭代问题）"""
        model.eval()

        logits = model(*X)
        loss = unspv_loss_func(logits, Y)
        loss.backward()

        # 修复：兼容模型可能没有set_running_stat_grads方法的情况
        if hasattr(model, 'set_running_stat_grads'):
            model.set_running_stat_grads()

        # ========== 核心修复：正确获取可训练参数和梯度 ==========
        trainable_params = self.get_trainable_params(model)

        # 获取梯度（增加梯度存在性检查和空值处理）
        unspv_grad = []
        for p in trainable_params:
            if p.grad is not None:
                unspv_grad.append(p.grad.clone())
            else:
                unspv_grad.append(torch.zeros_like(p, device=self.device))

        # 适配率更新（确保长度匹配）
        with torch.no_grad():
            # 严格校验长度
            if len(unspv_grad) != len(adapt_lrs):
                raise ValueError(f"梯度长度{len(unspv_grad)}与适配率长度{len(adapt_lrs)}不匹配！"
                                 f"\n可训练参数数量：{len(trainable_params)}"
                                 f"\n适配率数量：{len(adapt_lrs)}")

            # 正确迭代可训练参数（修复核心迭代问题）
            for i, (p, g) in enumerate(zip(trainable_params, unspv_grad)):
                if p.requires_grad:
                    p -= adapt_lrs[i] * g

        model.zero_grad()

        # 修复：兼容模型可能没有clip_bn_running_vars方法的情况
        if hasattr(model, 'clip_bn_running_vars'):
            model.clip_bn_running_vars()

        return unspv_grad

    def local_train(self, model, adapt_lrs, args, dataset='test'):
        """本地训练（增加鲁棒性处理）"""
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset] if hasattr(self, 'num_data') and dataset in self.num_data else len(
            dataloader.dataset)

        state = deepcopy(model.state_dict())

        for *X, Y in dataloader:
            model.load_state_dict(state)

            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 1. unsupervised adaptation
            unspv_grad = self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation
            model.eval()

            logits = model(*X)
            spv_loss = spv_loss_func(logits, Y)

            # 获取监督梯度（使用统一的参数获取方法）
            trainable_params = self.get_trainable_params(model)
            try:
                spv_grad = torch.autograd.grad(spv_loss, trainable_params)
            except RuntimeError as e:
                print(f"警告：梯度计算失败 - {e}，使用0梯度")
                spv_grad = [torch.zeros_like(g, device=self.device) for g in unspv_grad]

            # 3. update the adaptation rate
            with torch.no_grad():
                # 确保梯度长度匹配
                if len(spv_grad) != len(unspv_grad):
                    print(f"警告：监督梯度长度{len(spv_grad)}与无监督梯度长度{len(unspv_grad)}不匹配，使用0梯度")
                    spv_grad = [torch.zeros_like(g, device=self.device) for g in unspv_grad]

                # 初始化梯度更新值
                g = torch.zeros_like(adapt_lrs, device=self.device)

                # 梯度归一化逻辑（增加除零保护）
                if args.grad_norm == 'none':
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()

                elif args.grad_norm == 'numel':
                    l = torch.zeros_like(adapt_lrs, device=self.device)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()
                    g /= l.clamp(min=1)  # 防止除零

                elif args.grad_norm == 'sqrt_numel':
                    l = torch.zeros_like(adapt_lrs, device=self.device)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()
                    g /= torch.sqrt(l.clamp(min=1))  # 防止除零和开方负数

                elif args.grad_norm == 'manual':
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_params:
                            g[i] += (g1 * g2).sum()
                        elif i in args.idx_stats:
                            g[i] += 100 * (g1 * g2).sum()

                elif args.grad_norm == 'params_only':
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_params:
                            g[i] += (g1 * g2).sum()
                            g[i] /= max(g1.numel(), 1)  # 防止除零

                elif args.grad_norm == 'stats_only':
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_stats:
                            g[i] += (g1 * g2).sum()
                            g[i] /= max(g1.numel(), 1)  # 防止除零

                else:
                    raise NotImplementedError(f"不支持的梯度归一化方式：{args.grad_norm}")

                model_type = args.model

                # 更新适配率
                if 'resnet' in model_type or 'vit' in model_type:
                    adapt_lrs += self.lr * g

            # 记录损失和准确率
            with torch.no_grad():
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        # 防止除零
        avg_loss = total_loss / max(total_examples, 1)
        avg_metric = total_metric / max(total_examples, 1)

        return avg_loss, avg_metric, num_data

    def local_eval(self, model, adapt_lrs, args, dataset='test'):
        """本地评估（彻底修复迭代问题）"""
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset] if hasattr(self, 'num_data') and dataset in self.num_data else len(
            dataloader.dataset)

        state = deepcopy(model.state_dict())

        for i, (*X, Y) in enumerate(dataloader):
            model.load_state_dict(state)

            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 1. unsupervised adaptation
            self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation
            model.eval()

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        # 防止除零
        avg_loss = total_loss / max(total_examples, 1)
        avg_metric = total_metric / max(total_examples, 1)

        return avg_loss, avg_metric, num_data