import os
import pickle

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


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """计算softmax分布的熵"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def cc_loss(outputs, partialY, temp):
    """计算部分标签损失"""
    sm_outputs = F.softmax(outputs / temp, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


class ATPTestPasleOServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        # 初始化训练和测试客户端
        self.train_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # 加载预训练模型
        self.model = create_model(args)
        self.model.change_bn(mode='grad')
        self.model.eval()

        # 加载适应率
        self.adaptation_rates = self.load_adapt_lrs(args)

        # 加载类别阈值先验知识
        self.class_thresholds = self.load_class_thresholds(args)



        # PASLE算法参数（使用单一阈值）
        # self.thresh = 0.6  # 初始阈值
        # self.thresh_gap = 0.1  # 阈值下降范围
        # self.thresh_des = 0.001  # 每次下降步长
        # self.temp = 3  # 温度参数
        # self.buffer_size = 5  # 缓冲区大小
        # self.samples_buffer = None  # 样本缓冲区
        #
        # args.pasle_thresh = getattr(args, 'pasle_thresh', 0.6)  # 初始阈值
        # args.pasle_thresh_gap = getattr(args, 'pasle_thresh_gap', 0.1)  # 阈值下降幅度
        # args.pasle_thresh_des = getattr(args, 'pasle_thresh_des', 0.001)  # 阈值衰减率
        # args.pasle_temp = getattr(args, 'pasle_temp', 3)  # 温度参数
        # args.pasle_buffer_size = getattr(args, 'pasle_buffer_size', 5)  # 样本缓冲区大小
        # self.samples_buffer = None  # 样本缓冲区

        # # 新增：设置类别数（默认为10，以CIFAR-10为例）
        # args.num_classes = getattr(args, 'num_classes', 100)


    def load_class_thresholds(self, args):
        """从.pkl文件加载thresholds张量"""
        try:
            with open(args.class_threshold_path, 'rb') as f:
                thresholds = pickle.load(f)
            return thresholds.to(args.device)  # 加载后转移到指定设备
        except:
            # 加载失败时返回均匀阈值
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            return torch.ones(num_classes, device=args.device) * 0.5

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
        """运行服务器，直接进行测试"""
        self.adapt_and_eval(args, 'test')

    def adapt_and_eval(self, args, mode='test'):
        """适应和评估"""
        global_state = deepcopy(self.model.updated_state_dict())
        weights, losses, metrics = [], [], []

        clients = self.train_clients if mode == 'valid' else self.test_clients

        for cid, client in tqdm(clients.items()):
            # 使用改进的PASLE算法进行本地评估（传入单一阈值）
            loss, metric, num_data = client.local_eval_pasle(
                self.model,
                self.adaptation_rates,
                args,
                'test'
            )

            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # 重置模型状态
            self.model.load_state_dict(global_state, strict=False)

            # 更新阈值
            # if self.thresh > self.thresh_end:
            #     self.thresh -= self.thresh_des

        # 计算加权平均损失和指标
        agg_loss = sum(w * l for w, l in zip(weights, losses)) / sum(weights)
        agg_metric = sum(w * m for w, m in zip(weights, metrics)) / sum(weights)

        tqdm.write(f'\t Eval: Loss: {agg_loss:.4f} \t Metric: {agg_metric:.4f}')

        # 记录历史
        log_dict = {
            mode + '_losses': losses,
            mode + '_metrics': metrics,
            mode + '_wavg_loss': agg_loss,
            mode + '_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class ATPTestClient(BaseClient):
    def __init__(self, cid, datasets, args):
        super().__init__(cid, datasets, args)
        self.samples_buffer = None
        self.thresh = args.pasle_thresh

        self.thresh_des = args.pasle_thresh_des
        self.temp = args.pasle_temp
        self.buffer_size = args.pasle_buffer_size
        self.num_classes = args.num_classes  # 从参数中获取类别数
        self.thresh_gap = args.pasle_thresh_gap
        self.thresh_end = self.thresh - self.thresh_gap
        #self.adapt_lr = 1e-4

    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
        """单步适应"""
        model.eval()
        logits = model(*X)
        loss = unspv_loss_func(logits, Y)
        loss.backward()
        model.set_running_stat_grads()

        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                p -= adapt_lrs[i] * g

        model.zero_grad()
        model.clip_bn_running_vars()
        return unspv_grad

    def local_eval_pasle(self, model, adapt_lrs, args, dataset):
        """使用PASLE算法进行本地评估"""
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        current_lrs = adapt_lrs.clone()
        total_examples, total_loss, total_metric = 0, 0, 0
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]
        state = deepcopy(model.state_dict())

        for i, (*X, Y) in enumerate(dataloader):
            # 根据测试模式调整学习率和模型状态
            if args.test == 'batch':
                model.load_state_dict(state)
            elif args.test == 'large_batch':
                model.load_state_dict(state)
                current_lrs = adapt_lrs
            elif args.test == 'online_raw':
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
                #state = deepcopy(state_start)
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
                state_start = wavg_state(state, state_now, 1 / (i + 1))
                model.load_state_dict(state_start)
                current_lrs = adapt_lrs / (i + 1)
            elif args.test == 'online_au':
                # state_now = model.state_dict()
                # state_start = wau_state(state, state_now, 0.5)
                # model.load_state_dict(state_start)
                # state = deepcopy(state_start)

                state_now = model.state_dict()

                # 关键修复：先将X转移到模型所在设备（self.device）
                X_device = [x.to(self.device) for x in X]  # 临时转移，不影响后续数据处理

                # 计算当前批次的熵或不确定性（使用转移后的X_device）
                with torch.no_grad():
                    logits = model(*X_device)  # 此时输入与模型设备一致
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                # 基于熵动态调整α：高熵→激进，低熵→保守
                alpha = torch.sigmoid(entropy * 2 - 3).item()  # 将熵映射到[0.1, 0.9]
                alpha = max(0.1, min(0.9, alpha))
                print(alpha)

                state_start = wau_state(state, state_now, alpha)
                model.load_state_dict(state_start)
                state = deepcopy(state_start)

                #current_lrs = adapt_lrs * 0.9
            elif args.test == 'online_avg':
                if i == 0:
                    acc_state = deepcopy(state)
                else:
                    acc_state = deepcopy(model.state_dict())
                    model.load_state_dict(state)

            # 获取数据
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 使用改进的PASLE自适应过程
            self.adapt_with_pasle(
                model=model,
                adapt_lrs=current_lrs,
                X=X,
                Y=Y,
                unspv_loss_func=unspv_loss_func,
                args=args,
                thresh=self.thresh,
                thresh_gap=self.thresh_gap,
                thresh_des=self.thresh_des,
                temp=self.temp,
                buffer_size=self.buffer_size
            )

            # 2. 监督评估
            model.eval()
            if args.test == 'online_avg':
                state_now = model.state_dict()
                state_new = wavg_state(acc_state, state_now, i/(i+1))
                model.load_state_dict(state_new)

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)

                # 记录损失和准确率
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
        return avg_loss, avg_metric, num_data

    def adapt_with_pasle(self, model, adapt_lrs, X, Y, unspv_loss_func, args,
                         thresh, thresh_gap, thresh_des, temp, buffer_size):
        """PASLE算法的适应过程（使用单一阈值）"""
        model.eval()

        # 1. 记录原始输入样本数量
        origin_sample_num = X[0].shape[0]

        # 2. 合并当前输入和缓冲区数据
        if self.samples_buffer is not None:
            X = [torch.cat([x, buf], dim=0) for x, buf in zip(X, self.samples_buffer)]

        # 前向传播
        logits = model(*X)
        probs = F.softmax(logits, 1)
        probs_des, _ = torch.sort(probs, descending=True)

        # 计算置信度边界（使用单一阈值）
        margins = probs_des[:, 0] - probs_des[:, 1]
        mask_hard = margins > thresh  # 高置信度样本
        mask_unselect = (probs_des[:, 0] - probs_des[:, -1]) < thresh  # 低置信度样本
        mask_partial = ~ (mask_hard | mask_unselect)  # 中等置信度样本

        # 更新样本缓冲区
        if mask_unselect.any():
            _, idxs = torch.sort(margins[mask_unselect], descending=True)
            if idxs.shape[0] > buffer_size:
                self.samples_buffer = [x[mask_unselect][idxs][:buffer_size] for x in X]
            else:
                self.samples_buffer = [x[mask_unselect] for x in X]

        # 为部分样本生成部分标签（使用单一阈值）
        partial_labels = ((probs[mask_partial] + thresh) >
                         probs_des[mask_partial][:, 0].reshape(-1, 1)).long()

        # 计算损失
        if mask_hard.any():
            loss_hard = nn.CrossEntropyLoss()(logits[mask_hard] / temp,
                                            logits[mask_hard].detach().argmax(1))
        else:
            loss_hard = torch.tensor(0.0, device=logits.device)

        if mask_partial.any():
            loss_partial = cc_loss(logits[mask_partial], partial_labels.detach(), temp)
        else:
            loss_partial = torch.tensor(0.0, device=logits.device)



        # 加权损失
        total_samples = mask_hard.long().sum() + mask_partial.long().sum()
        if total_samples > 0:
            lam_hard = mask_hard.long().sum() / total_samples
            loss = loss_hard * lam_hard + loss_partial * (1 - lam_hard)
        else:
            loss = torch.tensor(0.0, device=logits.device)

        logits1 = logits[:origin_sample_num]
        loss2 = unspv_loss_func(logits1, Y)

        # print("ziixn")
        # print(mask_hard.long().sum())
        # print("buziixn")
        # print(mask_partial.long().sum())

        loss3 = loss
        # 反向传播和参数更新
        if loss3.requires_grad:
            loss3.backward()
            model.set_running_stat_grads()

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

        # # 更新阈值
        if self.thresh > self.thresh_end:
            self.thresh -= self.thresh_des

def wavg_state(state1, state2, lamda):
    """状态加权平均"""
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = lamda * state1[k] + (1 - lamda) * state2[k]
    return state

def wau_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = state1[k] + lamda * (state2[k]-state1[k])
    return state