import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from model.MyBatchNorm2d import MyBatchNorm2d
from utils import pickle_load

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class ATPTestServer(BaseServer):


    def __init__(self, train_datasets, test_datasets, args):

        BaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
        # which has identical behavior, but support taking gradient
        self.model.eval()

        self.adaptation_rates = self.load_adapt_lrs(args)


    def load_adapt_lrs(self, args):
        path = args.load_adapt_path
        idx = args.load_adapt_idx
        rnd = args.load_adapt_round

        if path == 'manual':
            rate = torch.zeros(102).to(args.device)
            lr = args.lm_lr
            m = args.batchadapt_bn_momentum

            # stats_idx = [] # 1, 2, 6, 7, ..., 96, 97
            # for i in range(20):
            #     stats_idx.append(i * 5 + 1)
            #     stats_idx.append(i * 5 + 2)

            if args.layers_to_adapt == 'none':
                pass

            elif args.layers_to_adapt == 'const':
                rate = torch.ones(102).to(args.device) * lr

            elif args.layers_to_adapt == 'first_conv_bn':
                params_idxs = [0, 3, 4]
                stats_idxs = [1, 2]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block1':
                params_idxs = [5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 23, 24]
                stats_idxs = [6, 7, 11, 12, 16, 17, 21, 22]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block2':
                params_idxs = [25, 28, 29, 30, 33, 34, 35, 38, 39, 40, 43, 44, 45, 48, 49]
                stats_idxs = [26, 27, 31, 32, 36, 37, 41, 42, 46, 47]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block3':
                params_idxs = [50, 53, 54, 55, 58, 59, 60, 63, 64, 65, 68, 69, 70, 73, 74]
                stats_idxs = [51, 52, 56, 57, 61, 62, 66, 67, 71, 72]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block4':
                params_idxs = [75, 78, 79, 80, 83, 84, 85, 88, 89, 90, 93, 94, 95, 98, 99]
                stats_idxs = [76, 77, 81, 82, 86, 87, 91, 92, 96, 97]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'last_layer':
                params_idxs = [100, 101]
                for idx in params_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'all_bn':
                params_idxs = []  # 3, 4, 8, 9, ..., 98, 99
                for i in range(20):
                    params_idxs.append(5 * i + 3)
                    params_idxs.append(5 * i + 4)

                stats_idxs = []  # 1, 2, 6, 7, ..., 96, 97
                for i in range(20):
                    stats_idxs.append(i * 5 + 1)
                    stats_idxs.append(i * 5 + 2)

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'all_bn_stats':

                stats_idxs = []  # 1, 2, 6, 7, ..., 96, 97
                for i in range(20):
                    stats_idxs.append(i * 5 + 1)
                    stats_idxs.append(i * 5 + 2)

                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'all_bn_running_mean':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 1)

                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'all_bn_running_var':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 2)

                for idx in stats_idxs:
                    rate[idx] = m


            elif args.layers_to_adapt == 'all_bn_weight':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 3)

                for idx in stats_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'all_bn_bias':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 4)

                for idx in stats_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'all_conv':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5)

                for idx in stats_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'last_weight':
                params_idxs = [100, ]
                for idx in params_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'last_bias':
                params_idxs = [101, ]
                for idx in params_idxs:
                    rate[idx] = lr



            print(rate)


        elif path == 'zero':
            rate = torch.zeros(102).to(args.device)

        else:
            data = pickle_load(path, True)[idx]
            rate = data['history']['adapt_lrs'][rnd]
            rate = torch.Tensor(rate).to(args.device)

        return rate


    def run(self, args):

        # No Training, Direct Evaluation
        # self.adapt_and_eval(args, 'valid')
        self.adapt_and_eval(args, 'test')

    def adapt_and_eval(self, args, mode='test'):
        # current global model
        global_state = deepcopy(self.model.updated_state_dict())

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        if mode == 'valid':
            clients = self.train_clients
        else:
            clients = self.test_clients

        for cid, client in tqdm(clients.items()):
            loss, metric, num_data = client.local_eval(self.model, self.adaptation_rates, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # reset the model (the adaptation rate is not update, do not need to reset)
            self.model.load_state_dict(global_state, strict=False)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            mode + '_losses': losses,
            mode + '_metrics': metrics,
            mode + '_wavg_loss': agg_loss,
            mode + '_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

class ATPTestClient(BaseClient):

    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):

        model.eval()

        logits = model(*X)

        loss = unspv_loss_func(logits, Y)

        loss.backward()

        model.set_running_stat_grads()

        # unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]
        #
        # with torch.no_grad():
        #     for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
        #         p -= adapt_lrs[i] * g
        #
        # model.zero_grad()
        #
        # model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.
        if callable(getattr(model, 'trainable_parameters', None)):
            # 如果是方法，执行并获取列表
            trainable_params = model.trainable_parameters()
        else:
            # 如果是列表，直接使用
            trainable_params = model.trainable_parameters

        unspv_grad = [p.grad.clone() for p in trainable_params]
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(trainable_params, unspv_grad)):
                if adapt_lrs[i] != 0:
                    p -= adapt_lrs[i] * g

        model.zero_grad()
        model.clip_bn_running_vars()

        return unspv_grad

    def local_eval(self, model, adapt_lrs, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        current_lrs = adapt_lrs.clone()

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        state = deepcopy(model.state_dict())

        for i, (*X, Y) in enumerate(dataloader):

            if args.test == 'batch':
                # Use the same global client for all batches
                model.load_state_dict(state)

            elif args.test == 'large_batch':
                # Use the same global client for all batches
                model.load_state_dict(state)
                current_lrs = adapt_lrs

            elif args.test == 'online_raw':
                # Directly pass the current model for next batch
                pass

            elif args.test == 'online_small':
                current_lrs = adapt_lrs / 10

            elif args.test == 'online_exp':
                current_lrs = adapt_lrs * (0.5 ** i) * 0.6667


            elif args.test == 'online':

                # state_now = model.state_dict()

                # state_start = wavg_state(state, state_now, 0.5)

                # model.load_state_dict(state_start)

                # current_lrs = adapt_lrs * 0.5

                # state_now = model.state_dict()

                # state_start = wavg_state(state, state_now, 0.5)

                # model.load_state_dict(state_start)

                # state = deepcopy(state_start)

                state_now = model.state_dict()

                if args.dataset == 'pacs_aug':

                    state_start = wavg_state(state, state_now, 0.9)

                elif args.dataset == 'tiny_imagenet':

                    state_start = wavg_state(state, state_now, 0.5)

                else:

                    state_start = wavg_state(state, state_now, 0.9)

                model.load_state_dict(state_start)

            elif args.test == 'online_ha':
                state_now = model.state_dict()
                state_start = wavg_state(state, state_now, 1 / (i+1))
                model.load_state_dict(state_start)

                current_lrs = adapt_lrs / (i + 1)

            elif args.test == 'online_au':
                state_now = model.state_dict()

                # 关键修复：先将X转移到模型所在设备（self.device）
                X_device = [x.to(self.device) for x in X]  # 临时转移，不影响后续数据处理

                # 计算当前批次的熵或不确定性（使用转移后的X_device）
                with torch.no_grad():
                    logits = model(*X_device)  # 此时输入与模型设备一致
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                # 基于熵动态调整α：高熵→激进，低熵→保守
                alpha = torch.sigmoid(entropy * 2 - np.log(args.num_classes)).item()  # 将熵映射到[0.1, 0.9]
                alpha = max(0.1, max(0.9, alpha))  # 钳制范围

                state_start = wau_state(state, state_now, alpha)
                model.load_state_dict(state_start)
                state = deepcopy(state_start)
                state_now = model.state_dict()
                state_start = wau_state(state, state_now, 0.9)
                model.load_state_dict(state_start)

            elif args.test == 'online_avg':

                if i == 0:
                    acc_state = deepcopy(state)  # the average of all previous state

                else:  # i > 0
                    acc_state = deepcopy(model.state_dict())  # the average of all previous state
                    model.load_state_dict(state)





            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 1. unsupervised adaptation

            self.adapt_one_step(model, current_lrs, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation

            model.eval()

            if args.test == 'online_avg':
                state_now = model.state_dict()
                state_new = wavg_state(acc_state, state_now, i / (i+1))
                model.load_state_dict(state_new)

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data


def wavg_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = lamda * state1[k] + (1 - lamda) * state2[k]

    return state


def wau_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = state1[k] + lamda * (state2[k]-state1[k])
    return state

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# from copy import deepcopy
#
# from model import create_model, create_loss, create_metric, create_optimizer
# from model.MyBatchNorm2d import MyBatchNorm2d
# from utils import pickle_load
#
# from .Base import BaseServer, BaseClient
# from .TTABase import TTABaseServer
#
#
# class ATPTestServer(BaseServer):
#     def __init__(self, train_datasets, test_datasets, args):
#         BaseServer.__init__(self, train_datasets, test_datasets, args)
#
#         self.train_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
#         self.test_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
#
#         # load a pre-trained model
#         self.model = create_model(args)
#
#         # 修复1：兼容ViT的LayerNorm和CNN的BatchNorm
#         if args.model == 'vit':
#             self.model.change_bn(mode='grad')  # ViT的change_bn适配LayerNorm
#         else:
#             self.model.change_bn(mode='grad')  # CNN替换BatchNorm
#         self.model.eval()
#
#         self.adaptation_rates = self.load_adapt_lrs(args)
#
#     def load_adapt_lrs(self, args):
#         path = args.load_adapt_path
#         idx = args.load_adapt_idx
#         rnd = args.load_adapt_round
#
#         if path == 'manual':
#             rate = torch.zeros(102).to(args.device)
#             lr = args.lm_lr
#             m = args.batchadapt_bn_momentum
#
#             if args.layers_to_adapt == 'none':
#                 pass
#
#             elif args.layers_to_adapt == 'const':
#                 rate = torch.ones(102).to(args.device) * lr
#
#             elif args.layers_to_adapt == 'first_conv_bn':
#                 params_idxs = [0, 3, 4]
#                 stats_idxs = [1, 2]
#
#                 for idx in params_idxs:
#                     rate[idx] = lr
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'block1':
#                 params_idxs = [5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 23, 24]
#                 stats_idxs = [6, 7, 11, 12, 16, 17, 21, 22]
#
#                 for idx in params_idxs:
#                     rate[idx] = lr
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'block2':
#                 params_idxs = [25, 28, 29, 30, 33, 34, 35, 38, 39, 40, 43, 44, 45, 48, 49]
#                 stats_idxs = [26, 27, 31, 32, 36, 37, 41, 42, 46, 47]
#
#                 for idx in params_idxs:
#                     rate[idx] = lr
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'block3':
#                 params_idxs = [50, 53, 54, 55, 58, 59, 60, 63, 64, 65, 68, 69, 70, 73, 74]
#                 stats_idxs = [51, 52, 56, 57, 61, 62, 66, 67, 71, 72]
#
#                 for idx in params_idxs:
#                     rate[idx] = lr
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'block4':
#                 params_idxs = [75, 78, 79, 80, 83, 84, 85, 88, 89, 90, 93, 94, 95, 98, 99]
#                 stats_idxs = [76, 77, 81, 82, 86, 87, 91, 92, 96, 97]
#
#                 for idx in params_idxs:
#                     rate[idx] = lr
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'last_layer':
#                 params_idxs = [100, 101]
#                 for idx in params_idxs:
#                     rate[idx] = lr
#
#             elif args.layers_to_adapt == 'all_bn':
#                 params_idxs = []  # 3, 4, 8, 9, ..., 98, 99
#                 for i in range(20):
#                     params_idxs.append(5 * i + 3)
#                     params_idxs.append(5 * i + 4)
#
#                 stats_idxs = []  # 1, 2, 6, 7, ..., 96, 97
#                 for i in range(20):
#                     stats_idxs.append(i * 5 + 1)
#                     stats_idxs.append(i * 5 + 2)
#
#                 for idx in params_idxs:
#                     rate[idx] = lr
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'all_bn_stats':
#                 stats_idxs = []  # 1, 2, 6, 7, ..., 96, 97
#                 for i in range(20):
#                     stats_idxs.append(i * 5 + 1)
#                     stats_idxs.append(i * 5 + 2)
#
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'all_bn_running_mean':
#                 stats_idxs = []
#                 for i in range(20):
#                     stats_idxs.append(i * 5 + 1)
#
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'all_bn_running_var':
#                 stats_idxs = []
#                 for i in range(20):
#                     stats_idxs.append(i * 5 + 2)
#
#                 for idx in stats_idxs:
#                     rate[idx] = m
#
#             elif args.layers_to_adapt == 'all_bn_weight':
#                 stats_idxs = []
#                 for i in range(20):
#                     stats_idxs.append(i * 5 + 3)
#
#                 for idx in stats_idxs:
#                     rate[idx] = lr
#
#             elif args.layers_to_adapt == 'all_bn_bias':
#                 stats_idxs = []
#                 for i in range(20):
#                     stats_idxs.append(i * 5 + 4)
#
#                 for idx in stats_idxs:
#                     rate[idx] = lr
#
#             elif args.layers_to_adapt == 'all_conv':
#                 stats_idxs = []
#                 for i in range(20):
#                     stats_idxs.append(i * 5)
#
#                 for idx in stats_idxs:
#                     rate[idx] = lr
#
#             elif args.layers_to_adapt == 'last_weight':
#                 params_idxs = [100, ]
#                 for idx in params_idxs:
#                     rate[idx] = lr
#
#             elif args.layers_to_adapt == 'last_bias':
#                 params_idxs = [101, ]
#                 for idx in params_idxs:
#                     rate[idx] = lr
#
#             print(rate)
#
#         elif path == 'zero':
#             rate = torch.zeros(102).to(args.device)
#
#         else:
#             data = pickle_load(path, True)[idx]
#             rate = data['history']['adapt_lrs'][rnd]
#             rate = torch.Tensor(rate).to(args.device)
#
#         return rate
#
#     def run(self, args):
#         # No Training, Direct Evaluation
#         self.adapt_and_eval(args, 'test')
#
#     def adapt_and_eval(self, args, mode='test'):
#         # 修复2：ViT模型没有updated_state_dict()方法，改用通用的state_dict()
#         global_state = deepcopy(self.model.state_dict())
#
#         weights = []  # weights (importance) for each client
#         losses = []  # local testing losses
#         metrics = []  # local testing metrics (accuracies)
#
#         if mode == 'valid':
#             clients = self.train_clients
#         else:
#             clients = self.test_clients
#
#         for cid, client in tqdm(clients.items()):
#             loss, metric, num_data = client.local_eval(self.model, self.adaptation_rates, args, 'test')
#             weights.append(num_data)
#             losses.append(loss)
#             metrics.append(metric)
#
#             # reset the model
#             self.model.load_state_dict(global_state, strict=False)
#
#         # eval loss and metric
#         agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
#         agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
#         tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))
#
#         log_dict = {
#             mode + '_losses': losses,
#             mode + '_metrics': metrics,
#             mode + '_wavg_loss': agg_loss,
#             mode + '_wavg_metric': agg_metric,
#         }
#         self.history.append(log_dict)
#
#
# class ATPTestClient(BaseClient):
#     def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
#         model.eval()
#
#         logits = model(*X)
#         loss = unspv_loss_func(logits, Y)
#         loss.backward()
#
#         # 修复3：兼容ViT的set_running_stat_grads（LayerNorm）
#         if hasattr(model, 'set_running_stat_grads'):
#             model.set_running_stat_grads()
#         else:
#             # ViT的LayerNorm手动设置梯度（兜底逻辑）
#             for m in model.modules():
#                 if isinstance(m, (nn.LayerNorm,)):
#                     if m.requires_grad:
#                         m.weight.grad = m.weight.grad if m.weight.grad is not None else torch.zeros_like(m.weight)
#                         m.bias.grad = m.bias.grad if m.bias.grad is not None else torch.zeros_like(m.bias)
#
#         # 修复4：trainable_parameters是属性（列表），不是方法，移除括号！
#         if hasattr(model, 'trainable_parameters'):
#             # 优先使用模型定义的trainable_parameters属性
#             trainable_params = model.trainable_parameters  # 关键：不加括号！
#         else:
#             # 兜底：获取所有可训练参数
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#
#         # 修复5：处理梯度为None的情况，避免clone()报错
#         unspv_grad = []
#         for p in trainable_params:
#             if p.grad is not None:
#                 unspv_grad.append(p.grad.clone())
#             else:
#                 unspv_grad.append(torch.zeros_like(p))
#
#         # 参数更新（适配adapt_lrs长度可能不足的情况）
#         with torch.no_grad():
#             for i, (p, g) in enumerate(zip(trainable_params, unspv_grad)):
#                 if i < len(adapt_lrs) and adapt_lrs[i] != 0:  # 防止索引越界
#                     p -= adapt_lrs[i] * g
#
#         model.zero_grad()
#
#         # 修复6：兼容ViT的clip_bn_running_vars
#         if hasattr(model, 'clip_bn_running_vars'):
#             model.clip_bn_running_vars()
#
#         return unspv_grad
#
#     def local_eval(self, model, adapt_lrs, args, dataset='test'):
#         unspv_loss_func = create_loss('ent')
#         spv_loss_func = create_loss('ce')
#         metric_func = create_metric('acc')
#
#         current_lrs = adapt_lrs.clone()
#
#         total_examples, total_loss, total_metric = 0, 0, 0
#
#         dataloader = self.dataloaders[dataset]
#         num_data = self.num_data[dataset]
#
#         state = deepcopy(model.state_dict())
#
#         for i, (*X, Y) in enumerate(dataloader):
#             if args.test == 'batch':
#                 model.load_state_dict(state)
#
#             elif args.test == 'large_batch':
#                 model.load_state_dict(state)
#                 current_lrs = adapt_lrs
#
#             elif args.test == 'online_raw':
#                 pass
#
#             elif args.test == 'online_small':
#                 current_lrs = adapt_lrs / 10
#
#             elif args.test == 'online_exp':
#                 current_lrs = adapt_lrs * (0.5 ** i) * 0.6667
#
#             elif args.test == 'online':
#                 state_now = model.state_dict()
#                 if args.dataset == 'pacs_aug':
#                     state_start = wavg_state(state, state_now, 0.9)
#                 elif args.dataset == 'tiny_imagenet':
#                     state_start = wavg_state(state, state_now, 0.5)
#                 else:
#                     state_start = wavg_state(state, state_now, 0.9)
#                 model.load_state_dict(state_start)
#
#             elif args.test == 'online_ha':
#                 state_now = model.state_dict()
#                 state_start = wavg_state(state, state_now, 1 / (i + 1))
#                 model.load_state_dict(state_start)
#                 current_lrs = adapt_lrs / (i + 1)
#
#             elif args.test == 'online_au':
#                 state_now = model.state_dict()
#                 # 数据设备对齐
#                 X_device = [x.to(self.device) for x in X]
#                 with torch.no_grad():
#                     logits = model(*X_device)
#                     probs = F.softmax(logits, dim=1)
#                     entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
#
#                 # 修复7：修正alpha钳制逻辑（原代码max(max(...))错误）
#                 alpha = torch.sigmoid(entropy * 2 - np.log(args.num_classes)).item()
#                 alpha = max(0.1, min(0.9, alpha))  # 正确的钳制范围：0.1~0.9
#
#                 state_start = wau_state(state, state_now, alpha)
#                 model.load_state_dict(state_start)
#                 state = deepcopy(state_start)
#                 state_now = model.state_dict()
#                 state_start = wau_state(state, state_now, 0.9)
#                 model.load_state_dict(state_start)
#
#             elif args.test == 'online_avg':
#                 if i == 0:
#                     acc_state = deepcopy(state)
#                 else:
#                     acc_state = deepcopy(model.state_dict())
#                     model.load_state_dict(state)
#
#             # 数据设备转移
#             X = [x.to(self.device) for x in X]
#             Y = Y.to(self.device)
#
#             # 1. unsupervised adaptation
#             self.adapt_one_step(model, current_lrs, X, Y, unspv_loss_func, args)
#
#             # 2. supervised evaluation
#             model.eval()
#
#             if args.test == 'online_avg':
#                 state_now = model.state_dict()
#                 state_new = wavg_state(acc_state, state_now, i / (i + 1))
#                 model.load_state_dict(state_new)
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
#         return avg_loss, avg_metric, num_data
#
#
# def wavg_state(state1, state2, lamda):
#     state = deepcopy(state1)
#     for k in state1.keys():
#         state[k] = lamda * state1[k] + (1 - lamda) * state2[k]
#     return state
#
#
# def wau_state(state1, state2, lamda):
#     state = deepcopy(state1)
#     for k in state1.keys():
#         state[k] = state1[k] + lamda * (state2[k] - state1[k])
#     return state