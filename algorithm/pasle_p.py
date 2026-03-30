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


class ATPPALSEPTestServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        # 初始化PASLE参数
        self._init_pasle_args(args)

        self.train_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
        # which has identical behavior, but support taking gradient
        self.model.eval()

        self.adaptation_rates = self.load_adapt_lrs(args)

        # # 加载类别阈值先验知识（用于损失加权）
        # self.class_thresholds = self.load_class_thresholds(args)

    def load_class_thresholds(self, args):
        """从.pkl文件加载类别阈值张量（用于损失加权）"""
        try:
            with open(args.class_threshold_path, 'rb') as f:
                thresholds = pickle.load(f)
            return thresholds.to(args.device)  # 加载后转移到指定设备
        except:
            # 加载失败时返回均匀阈值
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            return torch.ones(num_classes, device=args.device) * 0.5

    def _init_pasle_args(self, args):
        """初始化PASLE相关参数"""
        # 基本参数
        # args.use_pasle = getattr(args, 'use_pasle', True)  # 是否启用PASLE
        # args.pasle_thresh = getattr(args, 'pasle_thresh', 0.6)  # 初始阈值
        # args.pasle_thresh_gap = getattr(args, 'pasle_thresh_gap', 0.1)  # 阈值下降幅度
        # args.pasle_thresh_des = getattr(args, 'pasle_thresh_des', 0.001)  # 阈值衰减率
        # args.pasle_temp = getattr(args, 'pasle_temp',0.1)  # 温度参数
        # args.pasle_buffer_size = getattr(args, 'pasle_buffer_size', 5)  # 样本缓冲区大小
        # self.samples_buffer = None  # 样本缓冲区
        #
        # # 新增：设置类别数（默认为10，以CIFAR-10为例）
        # args.num_classes = getattr(args, 'num_classes', 100)  # 数据集类别数
        #
        # # 增强版PASLE_E相关参数
        # args.use_pasle_e = getattr(args, 'use_pasle_e', False)  # 是否启用PASLE_E
        # args.pasle_filter_k = getattr(args, 'pasle_filter_k', 100)  # 每个类筛选的样本数
        # args.pasle_lambda = getattr(args, 'pasle_lambda', 0.3)
        # # 客户端重置参数
        # args.reset_pasle_per_client = getattr(args, 'reset_pasle_per_client', False)  # 是否为每个客户端重置PASLE参数

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
    def __init__(self, cid, datasets, args):
        super().__init__(cid, datasets, args)

        # 初始化PASLE参数
        self.class_thresholds = self.load_class_thresholds(args)
        # self.class_thresholds = torch.ones(args.num_classes, device=args.device) * 0.6
        # self.class_thresholds1 = self.load_class_thresholds(args)


        #self.pasle_thresh = args.pasle_thresh
        self.pasle_thresh = self.class_thresholds.mean()

        self.pasle_thresh_end = self.pasle_thresh - args.pasle_thresh_gap
        self.class_thresholds_end = self.class_thresholds - args.pasle_thresh_gap
        # self.class_thresholds_end = torch.clamp(self.class_thresholds_end, min=0.1)  # 终止阈值不低于下限

        self.pasle_thresh_des = args.pasle_thresh_des
        self.pasle_temp = args.pasle_temp
        self.pasle_buffer_size = args.pasle_buffer_size
        self.samples_buffer = None
        #原始阈值缩放（放大）
        # self.class_thresholds = self.class_thresholds / 0.9  # 放大为原来的1/0.7≈1.43倍
        #
        # # 限制阈值在0.1-0.9之间
        # self.class_thresholds = torch.clamp(
        #     self.class_thresholds,
        #     min=0.1,  # 下限
        #     max=0.9  # 上限
        # )

        # 初始化PASLE_E相关参数
        # self.use_pasle_e = args.use_pasle_e
        # self.pasle_filter_k = args.pasle_filter_k
        # self.pasle_lambda = args.pasle_lambda
        # self.num_classes = args.num_classes  # 从参数中获取类别数

        # ---------------------- 新增：CDF映射所需参数 ----------------------
        self.entropy_window = []  # 存储每个批次的平均熵值（历史数据）
        self.window_size = 30  # 滑动窗口大小（可调整，建议20-50，平衡实时性与稳定性）
        self.num_classes = args.num_classes  # 类别数（窗口数据不足时的 fallback 映射）

        # # 如果使用PASLE_E，初始化原型相关参数
        # if self.use_pasle_e:
        #     self._init_prototype_params(args)
        #     # 加载类别阈值先验知识（用于损失加权）




    def load_class_thresholds(self, args):
        """从.pkl文件加载类别阈值张量（用于损失加权）"""
        try:
            with open(args.class_threshold_path, 'rb') as f:
                thresholds = pickle.load(f)
            return thresholds.to(args.device)  # 加载后转移到指定设备
        except:
            # 加载失败时返回均匀阈值
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            return torch.ones(num_classes, device=args.device) * 0.5

    def _init_prototype_params(self, args):
        """初始化PASLE_E的原型相关参数"""
        self.model_copy = None  # 用于特征提取的模型副本
        self.supports = None  # 原型支持集
        self.labels = None  # 原型标签
        self.ent = None  # 原型熵值
        self.num_classes = args.num_classes  # 类别数

    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
        model.eval()
        logits = model(*X)
        origin_sample_num = X[0].shape[0]

        # 1. 分别计算无监督损失和伪标签损失
        loss_ent = torch.tensor(0.0, device=self.device)
        loss_pasle = torch.tensor(0.0, device=self.device)

        # 计算无监督损失（熵最小化）
        if args.wujiandu:
            loss_ent = unspv_loss_func(logits, Y)
            if args.wujiandu_weighted:
                probs1 = F.softmax(logits, dim=1)
                pred1 = probs1.argmax(1)
                # entropy_loss = softmax_entropy(logits)

               #按类别加权
                w = 1 / self.class_thresholds.clone().detach() + self.pasle_thresh
                w = w / w.mean()
                loss_ent = (loss_ent * w[pred1]).mean()
            #loss_ent = entropy_loss


        # 计算伪标签损失（PASLE损失）
        if args.use_pasle or args.use_pasle_e:
            loss_pasle = self._compute_pasle_loss(model, logits, X, args,unspv_loss_func)

        # # 如果没有有效损失，直接返回
        # if loss_ent == 0 and loss_pasle == 0:
        #     return

        # # 2. 计算Pareto优化权重
        # model.zero_grad()
        # grads_ent = []
        # grads_pasle = []

        # if args.wujiandu and args.use_pasle:
        #
        #     loss_ent.backward(retain_graph=True)
        #     grads_ent = [p.grad.clone() for p in model.trainable_parameters()]
        #     model.zero_grad()
        #
        # # if args.use_pasle:
        #     loss_pasle.backward(retain_graph=True)
        #     grads_pasle = [p.grad.clone() for p in model.trainable_parameters()]
        #     model.zero_grad()

        if args.wujiandu and args.use_pasle:
            # 计算无监督损失的梯度
            if loss_ent != 0:
                loss_ent.backward(retain_graph=True)
                grads_ent = [p.grad.clone() for p in model.trainable_parameters()]
                model.zero_grad()
            else:
                grads_ent = [torch.zeros_like(p) for p in model.trainable_parameters()]

            # 计算伪标签损失的梯度
            if loss_pasle != 0:
                loss_pasle.backward(retain_graph=True)
                grads_pasle = [p.grad.clone() for p in model.trainable_parameters()]
                model.zero_grad()
            else:
                grads_pasle = [torch.zeros_like(p) for p in model.trainable_parameters()]

        # # 计算无监督损失的 Fisher 信息
        # if args.wujiandu and loss_ent != 0:
        #     loss_ent.backward(retain_graph=True)
        #     grads_ent = [p.grad.clone() for p in model.trainable_parameters()]
        #     fisher_ent = [(g ** 2).sum() for g in grads_ent]
        #     fisher_ent_sum = sum(fisher_ent)
        #     model.zero_grad()
        # else:
        #     fisher_ent_sum = torch.tensor(1e-8, device=self.device)
        #
        # # 计算伪标签损失的 Fisher 信息
        # if args.use_pasle and loss_pasle != 0:
        #     loss_pasle.backward(retain_graph=True)
        #     grads_pasle = [p.grad.clone() for p in model.trainable_parameters()]
        #     fisher_pasle = [(g ** 2).sum() for g in grads_pasle]
        #     fisher_pasle_sum = sum(fisher_pasle)
        #     model.zero_grad()
        # else:
        #     fisher_pasle_sum = torch.tensor(1e-8, device=self.device)
        #
        # # 使用 Fisher 信息计算权重（归一化）
        # total_fisher = fisher_ent_sum + fisher_pasle_sum
        # alpha = fisher_ent_sum / total_fisher  # 无监督损失的权重
        # beta = fisher_pasle_sum / total_fisher  # 伪标签损失的权重

        # 4. 动态权重（冲突感知）

        if args.use_pasle and args.wujiandu:
            def compute_conflict(g1, g2):
                g1_flat = torch.cat([g.flatten() for g in g1])
                g2_flat = torch.cat([g.flatten() for g in g2])
                return F.cosine_similarity(g1_flat.unsqueeze(0), g2_flat.unsqueeze(0), dim=1).item()

            cos_sim = compute_conflict(grads_ent, grads_pasle)
            #print(cos_sim)
            alpha = torch.sigmoid(torch.tensor(cos_sim))  # 放大差异
            #alpha = 0.1 + 0.8 * alpha  # 限制范围 [0.1, 0.9]
           # print(alpha)
            #alpha=0.9
            if args.wujiandu_first:

                total_loss = (1-alpha )*loss_pasle +  alpha*loss_ent
            else:
                total_loss = alpha * loss_pasle + (1-alpha ) * loss_ent
           #total_loss = alpha * loss_ent + beta * loss_pasle

        else:
            total_loss = loss_pasle + loss_ent


        # 3. 动态调整权重（Pareto优化）
        # if args.use_pasle and args.wujiandu:
        #     alpha = 0.5  # 初始权重
        #     cos_sim = 0.0
        #     cos_sims = []
        #     for g1, g2 in zip(grads_ent, grads_pasle):
        #         g1_flat = g1.flatten()
        #         g2_flat = g2.flatten()
        #         if torch.norm(g1_flat) > 0 and torch.norm(g2_flat) > 0:
        #             cos_sim += g1_flat.dot(g2_flat) / (torch.norm(g1_flat) * torch.norm(g2_flat))
        #
        #     # 如果梯度冲突（余弦相似度为负），调整权重
        #     # if cos_sim < 0:
        #     #     # 冲突时降低主导损失的权重（这里简化处理：固定比例调整）
        #     #     #alpha = max(0.1, alpha - 0.2)  # 权重下限0.1
        #     #     alpha = 1.0 / (1 + torch.exp(cos_sim))  # Sigmoid调整
        #     # else:
        #     #     # 一致时恢复均衡权重
        #     #     alpha = 0.5
        #
        #     # cos_sims.append(cos_sim)
        #     # avg_cos_sim = torch.mean(torch.stack(cos_sims))
        #     # probs = F.softmax(logits, dim=1)
        #     # entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
        #     # alpha = torch.sigmoid(entropy).item()
        #     # 4. 计算加权总损失
        #     # with torch.no_grad():
        #     #     prob = F.softmax(logits, dim=1)
        #     #     uncertainty = 1 - prob.max(dim=1)[0].mean()  # [0,1]越高越不确定
        #
        #     # 不确定性高时加强伪标签损失权重
        #     alpha = 0.8
        #     total_loss = alpha * loss_ent + (1 - alpha) * loss_pasle

        # 5. 反向传播和参数更新
        if total_loss.requires_grad:
            total_loss.backward()
            model.set_running_stat_grads()

            unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]
            with torch.no_grad():
                for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                    if adapt_lrs[i] != 0:
                        p -= adapt_lrs[i] * g


            model.zero_grad()
            model.clip_bn_running_vars()

    def _compute_pasle_loss(self, model, logits, X, args,unspv_loss_func):
        """计算PASLE损失（不执行反向传播）"""
        model.eval()
        origin_sample_num = X[0].shape[0]



        if self.samples_buffer is not None:
            # 对多输入模型，需要分别合并每个输入
            X = [torch.cat([x, buf], dim=0) for x, buf in zip(X, self.samples_buffer)]
        # 确保X[0]是输入图像




        if args.jun:

            logits = model(*X)

            # 计算预测概率
            probs = F.softmax(logits, dim=1)

            # 获取每个样本概率最高的前三类及其概率值和索引
            probs_des, indices_des = torch.sort(probs, descending=True)  # 按概率降序排序
            k=args.K
            top3_probs = probs_des[:, :k]  # 前三高概率
            top3_indices = indices_des[:, :k]  # 前三高概率对应的类别索引

            # 计算类别边际(最大概率 - 次大概率)
            margins = probs_des[:, 0] - probs_des[:, 1]

            # 获取前三类对应的边界阈值
            # 假设self.class_thresholds是一个包含所有类别的阈值张量
            top3_thresholds = self.class_thresholds[top3_indices]  # 形状为[batch_size, 3]

            # 计算每个样本的加权对比阈值(概率加权)
            weighted_thresholds = (torch.sum(top3_probs * top3_thresholds, dim=1)/ k) # 形状为[batch_size]

            # 计算最大概率与最小概率的差值(用于低置信度判断)
            max_min_diff = probs_des[:, 0] - probs_des[:, -1]

            # 使用加权阈值进行样本划分
            mask_hard = margins > weighted_thresholds  # 高置信度样本(超过加权阈值)
            mask_unselect = max_min_diff < weighted_thresholds  # 低置信度样本(低于加权阈值)
            mask_partial = ~(mask_hard | mask_unselect)  # 中等置信度样本(保留多标签)



            # print(f"bianjie阈值={margins}")
            # print(f"class_thresholds={class_specific_thresholds}")
            #样本筛选掩码
            # mask_hard = margins > self.pasle_thresh
            # mask_unselect = (probs_des[:, 0] - probs_des[:, -1]) < self.pasle_thresh
            # mask_partial = ~(mask_hard | mask_unselect)
        else:
            logits = model(*X)
            # samples = X[0]

            # 计算预测概率
            probs = F.softmax(logits, dim=1)
            probs_des, _ = torch.sort(probs, descending=True)
            margins = probs_des[:, 0] - probs_des[:, 1]  # 计算类别边际
            # print(f"bianjie阈值={margins}")

            # 计算每个样本的预测类别
            pred_classes = probs.argmax(dim=1)

            # 使用类别特定的阈值
            class_specific_thresholds = self.class_thresholds[pred_classes]
            margins = probs_des[:, 0] - probs_des[:, 1]
            mask_hard = margins > class_specific_thresholds  # 高置信度样本(使用类别特定阈值)
            mask_unselect = (probs_des[:, 0] - probs_des[:, -1]) < class_specific_thresholds  # 低置信度样本
            mask_partial = ~ (mask_hard | mask_unselect)  # 中等置信度样本

        # 筛选难样本存入缓冲区
        if mask_unselect.any():
            #print("111111111111111111111111111")
            _, idxs = torch.sort(margins[mask_unselect], descending=True)
            # 对X中的每个tensor分别应用mask和indexing
            if idxs.shape[0] > self.pasle_buffer_size:
                self.samples_buffer = [x[mask_unselect][idxs][:self.pasle_buffer_size] for x in X]
            else:
                self.samples_buffer = [x[mask_unselect] for x in X]

        # 生成部分标签

        # partial_labels = ((probs[mask_partial] + self.pasle_thresh) >
        #                   probs_des[mask_partial, 0].reshape(-1, 1)).long()
        #print(partial_labels)

        partial_labels = ((probs[mask_partial] + self.class_thresholds) >
                          probs_des[mask_partial, 0].reshape(-1, 1)).long()
        #print(partial_labels)

        loss_hard_ent=0
        loss_partial_ent=0

        # 计算PASLE损失
        if torch.sum(mask_hard) > 0:
            loss_hard = nn.CrossEntropyLoss()(logits[mask_hard] / self.pasle_temp,
                                              logits[mask_hard].detach().argmax(1))
            #loss_hard_ent = unspv_loss_func(logits[mask_hard], pred_classes)
        else:
            loss_hard = torch.tensor(0.0, device=logits.device)

        if torch.sum(mask_partial) > 0:

            loss_partial = self._compute_cc_loss(logits[mask_partial], partial_labels, self.pasle_temp)
            #loss_partial_ent = unspv_loss_func(logits[mask_partial], pred_classes)
        else:
            loss_partial = torch.tensor(0.0, device=logits.device)

        # print(mask_hard.long().sum())
        # print(mask_partial.long().sum())

        #w = class_thresholds+thresh
        if args.pasle_weight:
            #w = self.class_thresholds.clone().detach()  # [C]
            #w = -torch.exp(w)  # 指数放大，可换 A/B/D
            #w = 1/self.class_thresholds.clone().detach()
            w = 1/torch.exp(-1. * self.class_thresholds.clone().detach())
            # w=w / w.mean()
            #print(w)

            # 2. 高置信度样本：CE 按预测类别加权
            if mask_hard.any():
                logits_hard = logits[mask_hard] / self.pasle_temp
                pseudo_hard = logits_hard.detach().argmax(1)
                ce_raw = F.cross_entropy(logits_hard, pseudo_hard, reduction='none')
                weight_hard = w[pseudo_hard]  # 一一对应
                #print(pseudo_hard)
                loss_hard = (ce_raw * weight_hard).mean()
            else:
                loss_hard = torch.tensor(0.0, device=logits.device)

            # 3. 中置信度样本：CC 按候选类别平均加权
            if mask_partial.any():
                logits_partial = logits[mask_partial]
                # partial_labels = ((F.softmax(logits_partial / self.pasle_temp, 1) + self.pasle_thresh) >
                #                   F.softmax(logits_partial /self.pasle_temp, 1).max(dim=1, keepdim=True)[0]).long()

                # 候选类别平均阈值
                cand_mask = partial_labels.bool()  # [B, C]
                avg_th = (cand_mask * w).sum(1) / (cand_mask.sum(1))
                #print(avg_th)
                # print("222")
                # print(avg_th)
                # weight_partial = 1.0 / (avg_th)  # [B]
                # weight_partial = weight_partial / weight_partial.mean()
                #weightt_partial = cand_mask * self.class_thresholds
               # print(partial_labels)
                # print("111")
                # print(weight_partial)

                cc_raw = cc_loss(logits_partial, partial_labels.detach(), self.pasle_temp)
                loss_partial = (cc_raw * avg_th).mean()  # 已归一化，直接乘即可
            else:
                loss_partial = torch.tensor(0.0, device=logits.device)

            # # 加权组合损失
            # if torch.sum(mask_hard) > 0:
            #     lam_hard = torch.sum(mask_hard.long()).float() / (
            #             torch.sum(mask_hard.long()) + torch.sum(mask_partial.long()))
            #     pasle_loss = loss_hard * lam_hard + loss_partial * (1 - lam_hard)
            # else:
            #     pasle_loss = loss_partial

        total_samples = mask_hard.long().sum() + mask_partial.long().sum()
        if total_samples > 0:
            lam_hard = mask_hard.long().sum() / total_samples
            loss = loss_hard * lam_hard + loss_partial * (1 - lam_hard)
            #loss = loss_hard


            # coeff = ((1 / (torch.exp(-1. * self.class_thresholds.clone().detach()))))
            # loss = loss.mul(coeff)
            # loss = loss.mean
            #loss = loss * torch.exp(-1. * self.class_thresholds.clone().detach())
            #loss_ent = loss_hard_ent * lam_hard + loss_partial_ent * (1 - lam_hard)
        else:
            loss = torch.tensor(0.0, device=logits.device)

       # # 如果使用PASLE_E，添加原型损失
       #  if self.use_pasle_e and torch.sum(mask_partial) > 0:
       #      proto_loss = self._compute_prototype_loss(model, X[0], logits, partial_labels, args)
       #      #print(f"proto_loss: {proto_loss.item()}, weighted: {self.pasle_lambda * proto_loss.item()}")
       #      loss = proto_loss
       #  if self.use_pasle_e:
       #      proto_loss = self._compute_prototype_loss(model, X[0], logits, partial_labels, args)
       #      loss += proto_loss


        print("mask_hard:", mask_hard.sum().item())
        print("mask_partial:", mask_partial.sum().item())


        # # 动态调整阈值
        # if self.pasle_thresh > self.pasle_thresh_end:
        #     #self.pasle_thresh -= self.pasle_thresh_des
        #     self.pasle_thresh *= args.pasle_thresh_rate
        # print(f"当前阈值={self.pasle_thresh:.3f}")

        if (self.class_thresholds > self.class_thresholds_end).all():
            #self.pasle_thresh -= self.pasle_thresh_des
            #self.class_thresholds *= args.pasle_thresh_rate
            self.class_thresholds -= self.pasle_thresh_des
            #self.class_thresholds -= (self.pasle_thresh_des / (self.class_thresholds1 ** 2))
        #print(self.class_thresholds)

        return loss

        #eturn torch.tensor(0.0, device=logits.device)

    def _compute_cc_loss(self, outputs, partial_labels, temp):
        """计算交叉一致性损失（Cross-Consistency Loss）"""
        #w=torch.exp(-1. * self.class_thresholds.clone().detach())
        sm_outputs = F.softmax(outputs / temp, dim=1)
        final_outputs = sm_outputs * partial_labels
        average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
        # final_sum = final_outputs.sum(dim=1).clamp(min=0.1)
        # average_loss = -torch.log(final_sum).mean()
        return average_loss

    def _compute_prototype_loss(self, model, samples, logits, partial_labels, args):
        """计算PASLE_E的原型损失"""
        if self.model_copy is None:
            # 初始化模型副本和原型支持集
            self.model_copy = deepcopy(model).eval()
            self._init_prototype_supports(args)

        # 获取特征和预测
        with torch.no_grad():
            # 修正：使用get_featurizer()方法获取特征提取器
            features = self.model_copy.featurizer()(samples)
            probs = F.softmax(logits, dim=1)
            y_hat = torch.nn.functional.one_hot(probs.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(logits)

        # 更新原型支持集
        self.supports = self.supports.to(features.device)
        self.labels = self.labels.to(features.device)
        self.ent = self.ent.to(features.device)
        self.supports = torch.cat([self.supports, features])
        self.labels = torch.cat([self.labels, y_hat])
        self.ent = torch.cat([self.ent, ent])

        # 筛选高质量原型
        supports, labels = self._select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = supports.T @ labels  # 计算类别原型

        # 计算当前样本与原型的损失
        features_norm = torch.nn.functional.normalize(features, dim=1)
        proto_dist = features_norm @ torch.nn.functional.normalize(weights, dim=0)  # 样本到原型的距离
        proto_loss = F.kl_div(F.log_softmax(proto_dist, dim=1),
                              F.softmax(logits.detach(), dim=1),
                              reduction='batchmean')

        return proto_loss

    def _init_prototype_supports(self, args):
        """初始化原型支持集"""
        with torch.no_grad():
            # 使用模型最后一层的权重作为初始原型
            # 修改：正确访问ResNet18的最后一层
            if hasattr(self.model_copy, 'classifier') and hasattr(self.model_copy.classifier, 'fc'):
                # 如果模型确实有classifier.fc结构，使用它
                self.supports = self.model_copy.classifier.fc.weight.data
            else:
                # 对于标准ResNet18，使用fc层
                self.supports = self.model_copy.backbone.fc.weight.data

            # 修改：正确调用分类器
            if hasattr(self.model_copy, 'classifier') and hasattr(self.model_copy.classifier, 'fc'):
                warmup_prob = self.model_copy.classifier(self.supports)
            else:
                warmup_prob = self.model_copy.backbone.fc(self.supports)

            self.warmup_ent = softmax_entropy(warmup_prob)
            self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()

            # 初始化原型支持集
            self.supports = self.supports.data
            self.labels = self.warmup_labels.data
            self.ent = self.warmup_ent.data

    def _select_supports(self):
        """筛选高质量原型样本"""
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.pasle_filter_k

        if filter_K == -1:  # 不筛选，使用所有样本
            indices = torch.arange(len(ent_s), device=ent_s.device)
        else:
            indices = []
            indices1 = torch.arange(len(ent_s), device=ent_s.device)
            for i in range(self.num_classes):
                class_mask = (y_hat == i)
                if torch.sum(class_mask) > 0:
                    _, indices2 = torch.sort(ent_s[class_mask])  # 按熵值排序（熵越小越可靠）
                    indices.append(indices1[class_mask][indices2[:filter_K]])  # 取每个类熵最小的K个样本

            if indices:
                indices = torch.cat(indices)
            else:
                indices = torch.tensor([], device=ent_s.device, dtype=torch.long)

        # 更新支持集
        if len(indices) > 0:
            self.supports = self.supports[indices]
            self.labels = self.labels[indices]
            self.ent = self.ent[indices]

        return self.supports, self.labels

    def local_eval(self, model, adapt_lrs, args, dataset='test'):
        # 重置PASLE参数（可选，取决于是否需要为每个客户端重置）
        # if args.reset_pasle_per_client:
        #     self.pasle_thresh = args.pasle_thresh
        #     self.samples_buffer = None
        #
        #     if self.use_pasle_e:
        #         self.supports = None
        #         self.labels = None
        #         self.ent = None

        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        current_lrs = adapt_lrs.clone()

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        state = deepcopy(model.state_dict())
        state_prev = deepcopy(model.state_dict())  # 初始基准

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
                state_now = model.state_dict()
                state_start = wau_state(state, state_now, 0.9)
                model.load_state_dict(state_start)
                #state_prev = deepcopy(state_start)

                #current_lrs = adapt_lrs * 0.9

            elif args.test == 'online_au':
                # state_now = model.state_dict()
                # state_start = wau_state(state, state_now, 0.9)
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
                alpha = torch.sigmoid(entropy*2-np.log(args.num_classes) ).item()  # 将熵映射到[0.1, 0.9]
                print(alpha)
                #alpha = max(0.1, min(0.9, alpha))  # 钳制范围


                state_start = wau_state(state, state_now, alpha)
                model.load_state_dict(state_start)
                state = deepcopy(state_start)

                #current_lrs = adapt_lrs * 0.9

            # elif args.test == 'online_au':
            #     state_now = model.state_dict()
            #     # 1. 输入设备对齐（原有代码，不变）
            #     X_device = [x.to(self.device) for x in X]
            #     with torch.no_grad():
            #         logits = model(*X_device)
            #         probs = F.softmax(logits, dim=1)
            #         current_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()  # 当前批次熵值（转标量）
            #
            #     # 2. 维护熵值滑动窗口（新增核心逻辑）
            #     self.entropy_window.append(current_entropy)  # 加入当前熵值
            #     # if len(self.entropy_window) > self.window_size:  # 超过窗口大小则移除最早数据
            #     #     self.entropy_window.pop(0)
            #
            #     # 3. 计算CDF（累积分布函数，当前熵值在历史中的分位数）
            #     if len(self.entropy_window) >= 3:  # 窗口数据足够（避免初期统计偏差）
            #         sorted_entropies = sorted(self.entropy_window)  # 历史熵值排序
            #         # 统计"小于等于当前熵值"的历史数据个数 → 分位数 = 个数 / 总个数
            #         cdf = sum(1 for ent in sorted_entropies if ent <= current_entropy) / len(sorted_entropies)
            #     else:  # 窗口数据不足时，用理论最大值做fallback映射
            #         max_theory_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32)).item()
            #         cdf = min(1.0, current_entropy / max_theory_entropy)  # 避免超出[0,1]
            #
            #     # 4. CDF映射到 [0.1, 0.9]（新增核心逻辑）
            #     alpha = 0.1 + 0.8 * cdf  # 线性缩放：CDF=0→0.1，CDF=1→0.9，中间线性过渡
            #     alpha = max(0.1, min(0.9, alpha))  # 双重保险：避免极端值（如窗口初期的异常熵值）
            #
            #     # 5. 模型状态更新（原有代码，不变）
            #     state_start = wau_state(state, state_now, alpha)
            #     model.load_state_dict(state_start)
            #     state = deepcopy(state_start)

            elif args.test == 'online_ha':
                state_now = model.state_dict()
                state_start = wavg_state(state, state_now, 1 / (i + 1))
                model.load_state_dict(state_start)

                current_lrs = adapt_lrs / (i + 1)

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
                state_new = wavg_state(acc_state, state_now, i / (i + 1))
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
        state[k] = lamda * state2[k] + (1 - lamda) * state1[k]
    return state

def wau_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = state1[k] + lamda * (state2[k]-state1[k])
    return state


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div


def topk_cluster(feature, supports, scores, p, k=3):
    """基于Top-K近邻的聚类损失"""
    # p: outputs of model batch x num_class
    feature = F.normalize(feature, 1)
    supports = F.normalize(supports, 1)
    sim_matrix = feature @ supports.T  # B,M
    topk_sim_matrix, idx_near = torch.topk(sim_matrix, k, dim=1)  # batch x K
    scores_near = scores[idx_near].detach().clone()  # batch x K x num_class
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near) ** 2, -1)

    loss = -1.0 * topk_sim_matrix * diff_scores
    return loss.mean()


def cc_loss(outputs, partialY, temp):
    """交叉一致性损失"""
    sm_outputs = F.softmax(outputs / temp, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss