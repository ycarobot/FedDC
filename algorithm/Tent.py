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
# class TentServer(TTABaseServer):
#
#     def __init__(self, train_datasets, test_datasets, args):
#         TTABaseServer.__init__(self, train_datasets, test_datasets, args)
#
#         self.train_clients = {cid: TentClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
#         self.test_clients = {cid: TentClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
#
#         # load a pre-trained model (loading in main.py)
#         self.model = create_model(args)
#         # train mode, because tent optimizes the model to minimize entropy
#         self.model.train()
#         # disable grad, to (re-)enable only what tent updates
#         self.model.requires_grad_(False)
#         # configure norm for tent updates: enable grad + force batch statisics
#         for m in self.model.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.requires_grad_(True)
#                 # force use of batch stats in train and eval modes
#                 m.track_running_stats = False
#                 m.running_mean = None
#                 m.running_var = None
#
#                 # print('one')
#
#
# class TentClient(BaseClient):
#
#     def local_eval(self, model, args, dataset='test'):
#         model.eval()
#         unspv_loss_func = create_loss('ent')
#         spv_loss_func = create_loss('ce')
#         metric_func = create_metric('acc')
#         optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)
#
#         total_examples, total_loss, total_metric = 0, 0, 0
#
#         for *X, Y in self.dataloaders[dataset]:
#             # Get a batch of data
#             X = [x.to(self.device) for x in X]
#             Y = Y.to(self.device)
#
#             logits = model(*X)
#             loss = unspv_loss_func(logits, Y)
#             loss.backward()
#             # print(model.backbone.bn1.weight)
#             optimizer.step()
#             # print(model.backbone.bn1.weight)
#             # exit()
#             optimizer.zero_grad()
#
#             with torch.no_grad():
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
#         # exit()
#
#         return avg_loss, avg_metric, total_examples

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class TentServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: TentClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: TentClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)

        # ========== 修复核心：适配不同模型的归一化层 ==========
        # 根据模型类型选择要启用梯度的归一化层
        if args.model == 'vit':
            # ViT模型：启用LayerNorm的梯度
            for m in self.model.modules():
                if isinstance(m, nn.LayerNorm) or hasattr(m, 'meta_scale'):  # 兼容你的StandardLayerNormAdapter
                    m.requires_grad_(True)
        else:
            # CNN/ResNet模型：启用BatchNorm2d的梯度（原有逻辑）
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None


class TentClient(BaseClient):

    def local_eval(self, model, args, dataset='test'):
        # 修复：ViT模型需要保持train模式（LayerNorm的统计依赖训练模式）
        # model.eval()  # 注释掉原有eval()，改为根据模型类型设置
        if args.model != 'vit':
            model.eval()
        else:
            model.train()  # ViT保持train模式以启用LayerNorm的梯度更新

        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            logits = model(*X)
            loss = unspv_loss_func(logits, Y)

            # 安全检查：确保loss有梯度
            if loss.requires_grad is False:
                raise RuntimeError("Loss tensor does not require gradient! Check model parameter grad settings.")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, total_examples