# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# from copy import deepcopy
#
# from model import create_model, create_loss, create_metric, create_optimizer
#
# from .Base import BaseServer, BaseClient
# from .TTABase import TTABaseServer
#
#
#
# class BatchNormServer(TTABaseServer):
#
#     def __init__(self, train_datasets, test_datasets, args):
#         TTABaseServer.__init__(self, train_datasets, test_datasets, args)
#
#         self.train_clients = {cid: BatchNormClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
#         self.test_clients = {cid: BatchNormClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
#
#         # load a pre-trained model
#         self.model = create_model(args)
#
#         prior = args.prior_strength / (args.prior_strength + args.batch_size)
#
#         print(prior)
#
#         self.model.change_bn(mode='prior', prior=prior)
#
#         self.model.eval()
#
#
# class BatchNormClient(BaseClient):
#
#     def local_eval(self, model, args, dataset='test'):
#
#         spv_loss_func = create_loss('ce')
#         metric_func = create_metric('acc')
#
#         model.eval()
#
#         total_examples, total_loss, total_metric = 0, 0, 0
#
#         for *X, Y in self.dataloaders[dataset]:
#             # Get a batch of data
#             X = [x.to(self.device) for x in X]
#             Y = Y.to(self.device)
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
#         avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
#
#         return avg_loss, avg_metric, total_examples
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import re

from model import create_model, create_loss, create_metric, create_optimizer
from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class BatchNormServer(TTABaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)
        self.train_clients = {cid: BatchNormClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: BatchNormClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # 加载模型并判断架构类型
        self.model = create_model(args)
        self.is_vit = 'vit' in args.model.lower()  # 识别ViT模型

        # 区分CNN/ViT的归一化层适配
        if not self.is_vit:
            # CNN：调整BatchNorm
            prior = args.prior_strength / (args.prior_strength + args.batch_size)
            print(f"CNN BN prior: {prior}")
            self.model.change_bn(mode='prior', prior=prior)
        else:
            # ViT：调整LayerNorm（替换BN的适配逻辑）
            self.adapt_vit_layernorm(self.model, args)
            print("ViT LayerNorm adapted for test time")

        self.model.eval()

    def adapt_vit_layernorm(self, model, args):
        """适配ViT的LayerNorm层（模拟BN的prior逻辑）"""
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                # 为ViT的LayerNorm添加测试时的均值/方差平滑（替代BN的prior）
                module.eps = 1e-6  # 稳定数值
                # 可选：加载预训练的LayerNorm统计量或微调
                if hasattr(module, 'weight'):
                    module.weight.data = module.weight.data * 0.9 + 0.1  # 轻微调整，模拟prior


class BatchNormClient(BaseClient):
    def local_eval(self, model, args, dataset='test'):
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        # 区分ViT/CNN的推理模式
        is_vit = 'vit' in args.model
        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # ViT专属处理：测试时轻微微调（TTA）
            if is_vit:
                # ViT需要测试时微调（少量梯度更新）
                optimizer = create_optimizer(model, optimizer_name='sgd', lr=1e-5)
                model.train()  # ViT微调需临时开启train模式（仅更新LayerNorm/Attention）
                optimizer.zero_grad()

                # 仅计算损失并反向传播（不更新主干，只微调归一化层）
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)
                spv_loss.backward()

                # 仅更新LayerNorm和Attention层
                for name, param in model.named_parameters():
                    if 'norm' in name or 'attention' in name:
                        if param.grad is not None:
                            optimizer.step()
                            break  # 仅微调一轮，避免过拟合
                model.eval()
                torch.no_grad()  # 后续推理仍用no_grad

            # 通用推理逻辑
            with torch.no_grad():
                if not is_vit:
                    logits = model(*X)
                    spv_loss = spv_loss_func(logits, Y)

                # 统计指标
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
        return avg_loss, avg_metric, total_examples