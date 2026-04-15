import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import math
from model import create_model, create_loss, create_metric, create_optimizer
from .Base import BaseServer, BaseClient


class FedAwiServer(BaseServer):
    """
    Server of FedAvg with enhanced weighting strategies and dynamic client participation
    """

    def __init__(self, train_datasets, test_datasets, args):
        super(FedAwiServer, self).__init__(train_datasets, test_datasets, args)

        # 检查或设置超参数
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds
        self.weighting_method = 'adaptive'  # 使用自适应权重方法

        # 每轮通信选择客户端的比例
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # 初始化客户端
        self.train_clients = {cid: FedAUClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: FedAUClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # 模型
        self.model = create_model(args)

        # 用于自适应权重的变量
        self.participate_intervals = {cid: [] for cid in self.train_clients.keys()}  # 记录参与间隔
        self.not_participate_counts = {cid: 0 for cid in self.train_clients.keys()}  # 记录未参与次数
        self.k_value = 5  # 自适应权重参数

        # 初始化客户端参与概率
        self.participation_prob_init = self._init_participation_prob(args)  # 初始参与概率
        self.participation_prob_current = self.participation_prob_init.copy()  # 当前参与概率

    # def _init_participation_prob(self):
    #     """
    #     初始化客户端参与概率，基于数据分布
    #     """
    #     # 这里简化实现，实际应根据数据分布计算
    #     num_clients = len(self.train_clients)
    #     # 前半部分客户端高参与概率，后半部分低参与概率
    #     uniform_first_half = np.random.uniform(0, 1, num_clients // 2)
    #     uniform_second_half = np.random.uniform(0, 0.5, num_clients - num_clients // 2)
    #     participation_prob = np.concatenate([uniform_first_half, uniform_second_half])
    #     print(participation_prob)
    #     # 归一化到合理范围
    #     #participation_prob = np.clip(participation_prob, 0.01, 0.9)
    #     return {cid: prob for cid, prob in zip(self.train_clients.keys(), participation_prob)}

    def _init_participation_prob(self,args):
        """用 Dirichlet(α) × 固定权重向量 生成初始参与概率"""
        num_clients = len(self.train_clients)
        num_labels = args.num_classes
        alg, *params = args.partition.split('_')

        # partition
        if alg == 'step':
            num_major = int(params[0])
            alpha = float(params[1])
            alpha = 0.1


        elif alg == 'stratified':
            num_major = 2
            alpha = 0.1


        # 1. 每个客户端采样一个类别分布
        dirichlet_dists = np.random.dirichlet([alpha] * num_labels, size=num_clients)

        # 2. 固定权重：一半类别 0-1，一半类别 0-0.5
        half = num_labels // 2
        w = np.concatenate([np.random.uniform(0, 1, half),
                            np.random.uniform(0, 0.5, num_labels - half)])

        # 3. 内积得到初始概率
        p_list = np.clip(dirichlet_dists @ w, 0, 1)
        print('participation_prob_each_node:', p_list, flush=True)

        return {cid: float(p) for cid, p in zip(self.train_clients.keys(), p_list)}

    def update_participation_prob(self, rnd,args):
        """
        根据轮次更新参与概率（模拟周期性波动）
        """
        fluctuate_type = args.fluctuate_type  # 波动类型，可配置

        for cid in self.train_clients.keys():
            init_prob = self.participation_prob_init[cid]

            if fluctuate_type == 1:
                # 正弦波动 (0.3幅度)
                self.participation_prob_current[cid] = (0.3 * np.sin(2 * np.pi / 20 * rnd) + 0.7) * init_prob
            elif fluctuate_type == 2:
                # 正弦波动 (0.2幅度)
                self.participation_prob_current[cid] = (0.2 * np.sin(2 * np.pi / 20 * rnd) + 0.8) * init_prob
            elif fluctuate_type == 3:
                # 正弦波动 (0.1幅度)
                self.participation_prob_current[cid] = (0.1 * np.sin(2 * np.pi / 20 * rnd) + 0.9) * init_prob
            elif fluctuate_type == 4:
                # 带阈值的正弦波动
                new_prob = (0.5 * np.sin(2 * np.pi / 20 * rnd) + 0.5) * init_prob
                self.participation_prob_current[cid] = new_prob if new_prob >= 0.1 else 0
            elif fluctuate_type == 5:
                # 周期性切换 (每10轮)
                if (rnd // 10) % 2 == 1:
                    self.participation_prob_current[cid] = 0.4 * init_prob
                else:
                    self.participation_prob_current[cid] = init_prob
            else:
                # 无波动
                self.participation_prob_current[cid] = init_prob

            # 确保概率在[0,1]范围内
            self.participation_prob_current[cid] = np.clip(self.participation_prob_current[cid], 0, 1)

    def sample_clients(self):
        """
        基于参与概率采样客户端
        """
        selected_cids = []
        for cid, client in self.train_clients.items():
            if np.random.binomial(1, self.participation_prob_current[cid]) == 1:
                selected_cids.append(cid)

        # 确保至少选择一个客户端
        if not selected_cids:
            selected_cids = [np.random.choice(list(self.train_clients.keys()))]

        return selected_cids

    # def sample_clients(self):
    #     """
    #     改为类似代码2的按比例随机选择客户端
    #     """
    #     selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
    #     selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]
    #     return selected_cids

    def calculate_weights(self, selected_cids, num_data_list):
        """
        根据不同的权重计算方法计算聚合权重
        """
        weights = []

        if self.weighting_method == 'data_size':
            # 默认方法：按数据量加权
            weights = num_data_list

        elif self.weighting_method == 'known_prob':
            # 已知参与概率的加权方法
            for cid in selected_cids:
                prob = self.participation_prob_current[cid]  # 使用当前参与概率
                weights.append(1 / prob)

        elif self.weighting_method == 'adaptive':
            # 自适应加权方法
            for cid in selected_cids:
                if len(self.participate_intervals[cid]) > 0:
                    # 使用平均参与间隔作为权重
                    avg_interval = np.mean(self.participate_intervals[cid])
                    weights.append(avg_interval)
                else:
                    weights.append(1)  # 默认权重

        # # 归一化权重
        # weight_sum = sum(weights)
        # normalized_weights = [w / weight_sum for w in weights]

        return weights

    def update_participation_stats(self, selected_cids):
        """
        更新客户端参与统计信息
        """
        for cid in self.train_clients.keys():
            if cid in selected_cids:
                # 记录参与间隔
                self.participate_intervals[cid].append(self.not_participate_counts[cid] + 1)
                self.not_participate_counts[cid] = 0
            else:
                self.not_participate_counts[cid] += 1
                # 如果未参与次数超过阈值，记录间隔
                if self.not_participate_counts[cid] >= self.k_value:
                    self.participate_intervals[cid].append(self.not_participate_counts[cid])
                    self.not_participate_counts[cid] = 0

    def run(self, args):
        """
        Run the training and testing pipeline with dynamic participation
        """
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))

            # 更新参与概率
            self.update_participation_prob(rnd,args)

            # 训练
            self.train(self.model, args, rnd)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args, rnd):
        """
        训练一个通信轮次，使用基于参与统计信息的权重计算方法
        """
        # 当前全局模型
        global_state = deepcopy(model.updated_state_dict())

        next_state = None
        weights = []  # 客户端权重
        losses = []  # 训练损失
        metrics = []  # 训练指标
        num_data_list = []  # 各客户端数据量

        # 基于参与概率采样客户端
        selected_cids = self.sample_clients()

        # 更新参与统计信息
        self.update_participation_stats(selected_cids)

        # 计算自适应权重（基于参与间隔）
        weights = self.calculate_weights(selected_cids, num_data_list)

        if self.weighting_method == 'adaptive':
            # 遍历选中的客户端
            for cid, weight in zip(tqdm(selected_cids), weights):
                client = self.train_clients[cid]
                model.load_state_dict(global_state, strict=False)  # 从全局模型开始

                # 传递agg_weight到模型
                model.agg_weight = weight  # 关键修改点

                loss, metric, num_data = client.local_train(model, args, 'train')
                local_state = model.updated_state_dict()

                num_data_list.append(num_data)
                # print(num_data)
                losses.append(loss)
                metrics.append(metric)

                # 累积权重（使用参与统计信息计算的权重）

                weighted_local_state = deepcopy(local_state)

                for k in weighted_local_state.keys():
                    weighted_local_state[k] = torch.mul(local_state[k], weight)

                if next_state is None:

                    next_state = weighted_local_state

                else:

                    for k in next_state.keys():
                        next_state[k] += weighted_local_state[k]

                # if next_state is None:
                #     next_state = deepcopy(local_state)
                #     for k in next_state.keys():
                #         next_state[k] = torch.mul(local_state[k], num_data)
                # else:
                #     for k in next_state.keys():
                #         next_state[k] += torch.mul(local_state[k], num_data)

        else:
            # 遍历选中的客户端
            for cid in (tqdm(selected_cids)):
                client = self.train_clients[cid]
                model.load_state_dict(global_state, strict=False)  # 从全局模型开始

                loss, metric, num_data = client.local_train(model, args, 'train')
                local_state = model.updated_state_dict()

                num_data_list.append(num_data)
                losses.append(loss)
                metrics.append(metric)

                if next_state is None:
                    next_state = deepcopy(local_state)
                    for k in next_state.keys():
                        next_state[k] = torch.mul(local_state[k], num_data)
                else:
                    for k in next_state.keys():
                        next_state[k] += torch.mul(local_state[k], num_data)

        if self.weighting_method == 'data':
            # 计算加权损失和指标
            weight_sum = sum(num_data_list)
            # print(num_data_list)
            agg_loss = sum([w * l for w, l in zip(num_data_list, losses)]) / weight_sum
            agg_metric = sum([w * m for w, m in zip(num_data_list, metrics)]) / weight_sum
            tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))
            # 聚合模型
            for k in next_state.keys():
                next_state[k] = torch.div(next_state[k], weight_sum)
            model.load_state_dict(next_state)
        else:
            # 计算加权损失和指标
            weight_sum = sum(weights)
            # weight_sum1 = sum(num_data_list)
            # agg_loss = sum([w * l for w, l in zip(weights, losses)]) / weight_sum1
            # agg_metric = sum([w * m for w, m in zip(weights, metrics)]) / weight_sum1
            # tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))
            agg_loss = sum(losses) / len(selected_cids)
            agg_metric = sum(metrics) / len(selected_cids)
            tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))
            # 聚合模型
            for k in next_state.keys():
                next_state[k] = torch.div(next_state[k], weight_sum)
            model.load_state_dict(next_state)


        # 记录日志
        log_dict = {
            'train_selected_cids': selected_cids,
            'train_losses': losses,
            'train_metrics': metrics,
            'train_weights': weights,
            'train_wavg_loss': agg_loss,
            'train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    def eval_part(self, model, args):
        """
        Evaluate the global model with unseen data on participating client (source clients)
        """
        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.train_clients.items()):
            loss, metric, num_data = client.local_eval(model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # eval loss and metric
        weight_sum = sum(weights)
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / weight_sum
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / weight_sum
        tqdm.write('\t Eval (Part):  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_part_losses': losses,
            'test_part_metrics': metrics,
            'test_part_weights': weights,
            'test_part_wavg_loss': agg_loss,
            'test_part_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    def eval_unpart(self, model, args):
        """
        Evaluate the global model with unseen client (target clients)
        """
        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.test_clients.items()):
            loss, metric, num_data = client.local_eval(model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # eval loss and metric
        weight_sum = sum(weights)
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / weight_sum
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / weight_sum
        tqdm.write('\t Eval (Unpart):  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_unpart_losses': losses,
            'test_unpart_metrics': metrics,
            'test_unpart_weights': weights,
            'test_unpart_wavg_loss': agg_loss,
            'test_unpart_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class FedAUClient(BaseClient):
    """
    Client of FedAU (Federated Averaging with Adaptive Updates)
    """

    def __init__(self, cid, datasets, args):
        super(FedAUClient, self).__init__(cid, datasets, args)
        #self.participation_count = 0  # 参与次数计数
        self.prev_model = None  # 存储上一轮模型

    def local_train(self, model, args, dataset='train'):
        """
        本地训练，加入自适应插值
        """
        # 保存上一轮模型（如果是第一次参与，则保存当前模型）
        torch.cuda.empty_cache()
        current_state = deepcopy(model.updated_state_dict())
        if self.prev_model is None:
            self.prev_model = deepcopy(current_state)

        # 从模型中获取agg_weight（默认为1.0）
        agg_weight = getattr(model, 'agg_weight', 1.0)  # 关键修改点
        # print(agg_weight)

        # 插值系数计算
        interp_coeff = self._adjust_interpolation_coefficient(agg_weight, args)

        # # 模型插值：w = w + coeff * (w - w_prev)
        # interpolated_state = deepcopy(current_state)
        # for k in interpolated_state.keys():
        #
        #     interpolated_state[k] = current_state[k] + interp_coeff * (current_state[k] - self.prev_model[k])

        with torch.no_grad():  # 临时禁用梯度，减少内存占用
            interpolated_state = {}
            for k in current_state.keys():
                interpolated_state[k] = current_state[k] + interp_coeff * (current_state[k] - self.prev_model[k])

        # 加载插值后的模型
        model.load_state_dict(interpolated_state, strict=False)

        """
                Local Training with Adaptive Update Capabilities
                """
        # ======== Extract Hyperparameters ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs
        batch_size = args.batch_size

        # ======== Prepare for Training ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # xiugai
        if len(dataloader.dataset) < 2:
            raise ValueError(
                f"Client {self.cid} has insufficient data ({len(dataloader.dataset)} samples). Need at least 2 samples for BatchNorm training.")

            # 如果数据量小于batch size，使用完整数据作为一个batch
        batch_size = min(args.batch_size, len(dataloader.dataset))
        if batch_size < 2:
            batch_size = 2  # 强制batch size至少为2

        # 重新创建DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False  # 不要丢弃最后不足batch size的数据
        )

        # ======== Training ========
        model.train()
        # self.participation_count += 1  # Increment participation counter

        total_examples, total_loss, total_metric = 0, 0, 0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                # Drop the last batch if necessary
                if model.drop_last and len(Y) < batch_size:
                    continue

                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # Forward pass
                logits = model(*X)
                loss = loss_func(logits, Y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # Record metrics
                    num_examples = len(X[0])
                    total_examples += num_examples
                    total_loss += loss.item() * num_examples
                    metric = metric_func(logits, Y)
                    total_metric += metric.item() * num_examples

        avg_loss = total_loss / total_examples
        avg_metric = total_metric / total_examples

        # 更新上一轮模型
        self.prev_model = current_state
        # self.participation_count += 1
        torch.cuda.empty_cache()

        return avg_loss, avg_metric, num_data

    # def local_train(self, model, args, dataset='train'):
    #     """
    #     本地训练，使用全局模型减去本地模型的差值进行插值
    #     """
    #     # 保存当前全局模型状态
    #     global_state = deepcopy(model.state_dict())
    #
    #     # 如果是第一次训练，保存初始状态
    #     if not hasattr(self, 'prev_local_state'):
    #         self.prev_local_state = deepcopy(global_state)
    #
    #     # ======== 模型插值部分 ========
    #     # 计算差值：global_model - local_model
    #     delta_state = {}
    #     for k in global_state.keys():
    #         delta_state[k] = global_state[k] - self.prev_local_state[k]
    #
    #     # 从模型中获取聚合权重（默认为1.0）
    #     agg_weight = getattr(model, 'agg_weight', 1.0)
    #
    #     # 计算插值系数
    #     interp_coeff = self._adjust_interpolation_coefficient(agg_weight, args)
    #
    #     # 应用插值：w = global + coeff * (global - local)
    #     interpolated_state = deepcopy(global_state)
    #     for k in interpolated_state.keys():
    #         #interpolated_state[k] = global_state[k] + 0.1 * delta_state[k]
    #         interpolated_state[k] = global_state[k] + interp_coeff * delta_state[k]
    #
    #     # 加载插值后的模型
    #     model.load_state_dict(interpolated_state, strict=False)
    #
    #     # ======== 训练部分 ========
    #     loss_func = create_loss(args.loss)
    #     metric_func = create_metric(args.metric)
    #     optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
    #     num_epochs = args.lm_epochs
    #
    #     dataloader = self.dataloaders[dataset]
    #     num_data = self.num_data[dataset]
    #
    #     # 处理小批量数据情况
    #     batch_size = min(args.batch_size, len(dataloader.dataset))
    #     if batch_size < 2:
    #         batch_size = 2
    #
    #     dataloader = torch.utils.data.DataLoader(
    #         dataloader.dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         drop_last=False
    #     )
    #
    #     # 训练过程
    #     model.train()
    #     total_examples, total_loss, total_metric = 0, 0, 0
    #
    #     for epoch in range(num_epochs):
    #         for *X, Y in dataloader:
    #             if model.drop_last and len(Y) < batch_size:
    #                 continue
    #
    #             X = [x.to(self.device) for x in X]
    #             Y = Y.to(self.device)
    #
    #             # 前向传播
    #             logits = model(*X)
    #             loss = loss_func(logits, Y)
    #
    #             # 反向传播
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #             # 记录指标
    #             with torch.no_grad():
    #                 num_examples = len(X[0])
    #                 total_examples += num_examples
    #                 total_loss += loss.item() * num_examples
    #                 metric = metric_func(logits, Y)
    #                 total_metric += metric.item() * num_examples
    #
    #     # 更新本地模型状态
    #     self.prev_local_state = deepcopy(model.state_dict())
    #
    #     avg_loss = total_loss / total_examples
    #     avg_metric = total_metric / total_examples
    #
    #     return avg_loss, avg_metric, num_data

    def _adjust_interpolation_coefficient(self, agg_weight, args):
        """
        调整插值系数
        """
        base_coefficient = args.pre_train_init
        adjusted_coefficient = base_coefficient / agg_weight
        # return max(0, min(adjusted_coefficient, base_coefficient))
        return adjusted_coefficient
