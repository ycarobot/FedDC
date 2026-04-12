"""
FedAvg (Federated Averaging)

Reference:
    Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas:
    Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017: 1273-1282
Implementation:
    https://github.com/pliang279/LG-FedAvg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient


class FedAvgServer(BaseServer):
    """
    Server of FedAvg with enhanced participation mechanisms
    """

    def __init__(self, train_datasets, test_datasets, args):
        super(FedAvgServer, self).__init__(train_datasets, test_datasets, args)

        # check or set hyperparameters
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # init clients
        self.train_clients = {cid: FedAvgClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: FedAvgClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # model
        self.model = create_model(args)

        # for dynamic participation
        self.participation_prob_init = self._init_participation_prob(args)  # initial participation probabilities
        self.participation_prob_current = self.participation_prob_init.copy()  # current probabilities
        self.weighting_method = 'data_size'  # default weighting method

    # def _init_participation_prob(self):
    #     """
    #     Initialize client participation probabilities based on data distribution
    #     """
    #     num_clients = len(self.train_clients)
    #     # first half clients have higher participation probability, second half have lower
    #     uniform_first_half = np.random.uniform(0, 1, num_clients // 2)
    #     uniform_second_half = np.random.uniform(0, 0.5, num_clients - num_clients // 2)
    #     participation_prob = np.concatenate([uniform_first_half, uniform_second_half])
    #     print(participation_prob)
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
        Update participation probabilities based on round number (simulating periodic fluctuations)
        """
        fluctuate_type = args.fluctuate_type  # configurable fluctuation type

        for cid in self.train_clients.keys():
            init_prob = self.participation_prob_init[cid]

            if fluctuate_type == 1:
                # sine wave (0.3 amplitude)
                self.participation_prob_current[cid] = (0.3 * np.sin(2 * np.pi / 20 * rnd) + 0.7) * init_prob
            elif fluctuate_type == 2:
                # sine wave (0.2 amplitude)
                self.participation_prob_current[cid] = (0.2 * np.sin(2 * np.pi / 20 * rnd) + 0.8) * init_prob
            elif fluctuate_type == 3:
                # sine wave (0.1 amplitude)
                self.participation_prob_current[cid] = (0.1 * np.sin(2 * np.pi / 20 * rnd) + 0.9) * init_prob
            elif fluctuate_type == 4:
                # sine wave with threshold
                new_prob = (0.5 * np.sin(2 * np.pi / 20 * rnd) + 0.5) * init_prob
                self.participation_prob_current[cid] = new_prob if new_prob >= 0.1 else 0
            elif fluctuate_type == 5:
                # periodic switching (every 10 rounds)
                if (rnd // 10) % 2 == 1:
                    self.participation_prob_current[cid] = 0.4 * init_prob
                else:
                    self.participation_prob_current[cid] = init_prob
            else:
                # no fluctuation
                self.participation_prob_current[cid] = init_prob

            # ensure probability is within [0,1] range
            self.participation_prob_current[cid] = np.clip(self.participation_prob_current[cid], 0, 1)

    def sample_clients(self):
        """
        Sample clients based on participation probabilities
        """
        selected_cids = []
        for cid, client in self.train_clients.items():
            if np.random.binomial(1, self.participation_prob_current[cid]) == 1:
                selected_cids.append(cid)

        # ensure at least one client is selected
        if not selected_cids:
            selected_cids = [np.random.choice(list(self.train_clients.keys()))]

        return selected_cids

    def run(self, args):
        """
        Run the training and testing pipeline with dynamic participation
        """
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))

            # update participation probabilities
            self.update_participation_prob(rnd,args)

            # train
            self.train(self.model, args)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args):
        """
        Train for one communication round with dynamic client participation
        """
        # current global model
        global_state = deepcopy(model.updated_state_dict())

        next_state = None
        weights = []  # weights for each client
        losses = []  # training losses
        metrics = []  # training metrics
        num_data_list = []  # number of data points per client

        # sample clients based on participation probabilities
        selected_cids = self.sample_clients()

        # iterate randomly selected clients
        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            model.load_state_dict(global_state, strict=False)  # start from global model

            loss, metric, num_data = client.local_train(model, args, 'train')
            local_state = model.updated_state_dict()

            # Calculate weight for this client (inverse probability, not normalized)
            # weight = 1.0 / self.participation_prob_current[cid]
            #
            # weights.append(weight)

            num_data_list.append(num_data)

            losses.append(loss)
            metrics.append(metric)

            # accumulate weights (equal weighting)
            if next_state is None:
                next_state = deepcopy(local_state)
            else:
                for k in next_state.keys():
                    next_state[k] += local_state[k]

        # total_inv_prob = sum(1.0 / self.participation_prob_current[cid] for cid in selected_cids)

        # train loss and metric
        #weight_sum1 = sum(num_data_list)
        agg_loss = sum(losses) / len(selected_cids)
        agg_metric = sum(metrics) / len(selected_cids)
        tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))
        # agg_loss = sum([w * l for w, l in zip(weights, losses)]) / weight_sum1
        # agg_metric = sum([w * m for w, m in zip(weights, metrics)]) / weight_sum1
        # tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        # aggregate
        for k in next_state.keys():
            next_state[k] = torch.div(next_state[k], len(selected_cids))

        model.load_state_dict(next_state)

        log_dict = {
            'train_selected_cids': selected_cids,
            'train_losses': losses,
            'train_metrics': metrics,
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
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_part_losses': losses,
            'test_part_metrics': metrics,
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
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval(un part):  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_unpart_losses': losses,
            'test_unpart_metrics': metrics,
            'test_unpart_wavg_loss': agg_loss,
            'test_unpart_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class FedAvgClient(BaseClient):
    """
    Client of FedAvg
    """

    def __init__(self, cid, datasets, args):
        super(FedAvgClient, self).__init__(cid, datasets, args)

    def local_train(self, model, args, dataset='train'):
        """
        Local Training
        """

        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs
        batch_size = args.batch_size

        # ======== ======== Prepare for Training ======== ========
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

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:

                # Drop the last batch if necessary
                # e.g. when using batch norm
                if model.drop_last and len(Y) < batch_size:
                    continue

                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # record the loss and accuracy
                    num_examples = len(X[0])
                    total_examples += num_examples

                    total_loss += loss.item() * num_examples

                    metric = metric_func(logits, Y)
                    total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data
