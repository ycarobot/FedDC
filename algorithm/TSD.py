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


def softmax_kl_loss(input_logits, target_logits):
    """计算KL散度损失"""
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='none')


class TSDServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        # 初始化训练和测试客户端
        self.train_clients = {cid: ATPTestTSDClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPTestTSDClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # 加载预训练模型
        self.model = create_model(args)
        self.model.eval()

        # TSD算法参数
        self.hparams = {
            'filter_K': args.filter_K if hasattr(args, 'filter_K') else 100,
            'gamma': args.gamma if hasattr(args, 'gamma') else 1,
            'lr': args.lr if hasattr(args, 'lr') else 1e-6,
            'lam': args.lam if hasattr(args, 'lam') else 0.6,
            'cached_loader': False
        }

    def run(self, args):
        """运行服务器，直接进行测试"""
        self.adapt_and_eval(args, 'test')

    def adapt_and_eval(self, args, mode='test'):
        """适应和评估"""
        global_state = deepcopy(self.model.updated_state_dict())
        weights, losses, metrics = [], [], []

        clients = self.train_clients if mode == 'valid' else self.test_clients

        for cid, client in tqdm(clients.items()):
            # 使用TSD算法进行本地评估
            loss, metric, num_data = client.local_eval_tsd(
                self.model,
                args,
                'test',
                self.hparams
            )

            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # 重置模型状态
            self.model.load_state_dict(global_state, strict=False)

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


class ATPTestTSDClient(BaseClient):
    def __init__(self, cid, datasets, args):
        super().__init__(cid, datasets, args)

    def local_eval_tsd(self, model, args, dataset, hparams):
        """使用TSD算法进行本地评估"""
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        # 创建TSD算法实例
        tsd = TSD(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=hparams['lr']),
            lam=hparams['lam'],
            filter_K=hparams['filter_K'],
            steps=hparams['gamma'],
            episodic=False
        )

        total_examples, total_loss, total_metric = 0, 0, 0
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        for i, (*X, Y) in enumerate(dataloader):
            # 获取数据
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 使用TSD进行自适应和预测
            outputs = tsd(X[0])

            # 计算监督损失和指标
            spv_loss = spv_loss_func(outputs, Y)

            # 记录损失和准确率
            num_examples = len(X[0])
            total_examples += num_examples
            total_loss += spv_loss.item() * num_examples
            metric = metric_func(outputs, Y)
            total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
        return avg_loss, avg_metric, num_data


# 从代码2中复制TSD类
class TSD(nn.Module):
    """
    Test-time Self-Distillation (TSD)
    CVPR 2023
    """
    def __init__(self, model, optimizer, lam=0, filter_K=100, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer  # 已通过@property正确定义
        self.classifier = model.classifier  # 已通过@property正确定义（nn.Linear层）
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K

        # 关键修改：直接访问Linear层的weight，而非fc.weight
        warmup_supports = self.classifier.weight.data.detach()  # 修改此处
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)  # 保持不变

        # 后续代码保持不变
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.warmup_scores = F.softmax(warmup_prob, 1)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.scores = self.warmup_scores.data
        self.lam = lam

    def forward(self, x):
        z = self.featurizer(x)
        p = self.classifier(z)

        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        scores = F.softmax(p, 1)

        with torch.no_grad():
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.scores = self.scores.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
            self.scores = torch.cat([self.scores, scores])

            supports, labels = self.select_supports()
            supports = F.normalize(supports, dim=1)
            weights = (supports.T @ (labels))

        dist, loss = self.prototype_loss(z, weights.T, scores, use_hard=False)

        loss_local = self.topk_cluster(z.detach().clone(), supports, self.scores, p, k=3)
        loss += self.lam * loss_local
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return p

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        self.scores = self.scores[indices]

        return self.supports, self.labels

    def prototype_loss(self, z, p, labels=None, use_hard=False, tau=1):
        """原型损失函数"""
        z = F.normalize(z, 1)
        p = F.normalize(p, 1)
        dist = z @ p.T / tau
        if labels is None:
            _, labels = dist.max(1)
        if use_hard:
            labels = labels.argmax(1)
            loss = F.cross_entropy(dist, labels)
        else:
            loss = softmax_kl_loss(labels.detach(), dist).sum(1).mean(0)
        return dist, loss

    def topk_cluster(self, feature, supports, scores, p, k=3):
        """TopK聚类损失"""
        feature = F.normalize(feature, 1)
        supports = F.normalize(supports, 1)
        sim_matrix = feature @ supports.T  # B,M
        topk_sim_matrix, idx_near = torch.topk(sim_matrix, k, dim=1)  # batch x K
        scores_near = scores[idx_near].detach().clone()  # batch x K x num_class
        diff_scores = torch.sum((p.unsqueeze(1) - scores_near) ** 2, -1)
        loss = -1.0 * topk_sim_matrix * diff_scores
        return loss.mean()
