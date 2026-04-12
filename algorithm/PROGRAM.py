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


class PROGRAMServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        # 初始化训练和测试客户端
        self.train_clients = {cid: ATPTestPROGRAMClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPTestPROGRAMClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # 加载预训练模型
        self.model = create_model(args)
        self.model.eval()

        # 检查是否为ViT模型
        self.is_vit = hasattr(args, 'model') and 'vit' in args.model.lower()

        # PROGRAM算法参数（根据原论文设置默认值）
        self.hparams = {
            'lam': args.lam if hasattr(args, 'lam') else 0.5,  # 图传播系数
            'beta': args.beta if hasattr(args, 'beta') else 0.4,  # 非对称损失权重
            'lr': args.lr if hasattr(args, 'lr') else 1e-6,  # 优化器学习率
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
            # 使用PROGRAM算法进行本地评估
            loss, metric, num_data = client.local_eval_program(
                self.model,
                args,
                'test',
                self.hparams,
                self.is_vit  # 传递是否为ViT模型的标志
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


class ATPTestPROGRAMClient(BaseClient):
    def __init__(self, cid, datasets, args):
        super().__init__(cid, datasets, args)

    def local_eval_program(self, model, args, dataset, hparams, is_vit=False):
        """使用PROGRAM算法进行本地评估"""
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        # 创建PROGRAM算法实例，传递是否为ViT模型的标志
        program = PROGRAM(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=hparams['lr']),
            lam=hparams['lam'],
            beta=hparams['beta'],
            is_vit=is_vit  # 添加ViT标志
        )

        total_examples, total_loss, total_metric = 0, 0, 0
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        for i, (*X, Y) in enumerate(dataloader):
            # 获取数据
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 使用PROGRAM进行自适应和预测
            outputs = program(X[0])

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


class PROGRAM(nn.Module):
    """
    PROGRAM算法（来自代码2）
    基于图传播的测试时自适应方法
    支持ViT和CNN模型
    """
    def __init__(self, model, optimizer, lam=0.5, beta=0.4, is_vit=False):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer  # 特征提取器
        self.classifier = model.classifier  # 分类器
        self.optimizer = optimizer
        self.lam = lam  # 图传播系数
        self.beta = beta  # 非对称损失权重
        self.eps = 1e-4  # 数值稳定性参数
        self.is_vit = is_vit  # 是否为ViT模型

        # 初始化分类器权重作为图节点
        if hasattr(self.classifier, 'weight') and self.classifier.weight is not None:
            self.warmup_supports = self.classifier.weight.data.detach()
            self.num_classes = self.warmup_supports.size()[0]
            # 归一化权重作为初始支持向量
            self.warmup_supports = F.normalize(self.warmup_supports, p=2, dim=1)
            self.warmup_supports_T = self.warmup_supports.T.cuda()
        else:
            # 对于某些ViT变体，可能需要特殊处理
            self.warmup_supports = None
            self.num_classes = getattr(model, 'num_classes', 100)  # 默认100类
            print(f"Warning: Classifier has no weight attribute, using default {self.num_classes} classes")

    def extract_features(self, x):
        """提取特征，处理ViT和CNN的不同情况"""
        if self.is_vit:
            # ViT特征提取：需要处理不同的输出格式
            features = self.featurizer(x)
            # 如果特征是多维的，进行展平处理
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            return features
        else:
            # CNN特征提取
            return self.featurizer(x)

    def forward(self, x):
        # 提取特征和初始预测
        z = self.extract_features(x)  # 使用统一的特征提取方法
        p = self.classifier(z)  # 原始预测 [batch_size, num_classes]

        # 如果warmup_supports未初始化，使用随机初始化（针对ViT特殊情况）
        if self.warmup_supports is None:
            feature_dim = z.size(1)
            self.warmup_supports = torch.randn(self.num_classes, feature_dim).cuda()
            self.warmup_supports = F.normalize(self.warmup_supports, p=2, dim=1)
            self.warmup_supports_T = self.warmup_supports.T

        # 构建相似度图（公式1：特征与分类器权重的相似度）
        S = torch.matmul(z, self.warmup_supports_T)  # [batch_size, num_classes]

        # 构建图邻接矩阵（公式2：将分类器权重和输入特征拼接为图节点）
        Z = torch.cat((torch.eye(self.num_classes).cuda(), S), dim=0)  # [num_classes + batch_size, num_classes]

        # 图传播（公式3：计算归一化邻接矩阵）
        column_sums = torch.sum(Z, dim=0)  # 列求和用于归一化
        D = torch.diag(column_sums).cuda()  # 度矩阵
        D_inv = torch.inverse(D + self.eps * torch.eye(D.size(0)).cuda())  # 度矩阵的逆
        W = torch.matmul(Z, torch.matmul(D_inv, Z.T))  # 归一化邻接矩阵

        # 图传播后的伪标签（公式4：传播迭代）
        rows, cols = W.size()
        Y_star = (1 - self.lam) * torch.matmul(
            torch.inverse((torch.eye(rows, cols).cuda() - self.lam * W)),
            Z
        )  # [num_classes + batch_size, num_classes]
        # 取输入特征对应的伪标签并软化
        Y_tilde = F.softmax(Y_star[self.num_classes:, :], dim=1)  # [batch_size, num_classes]

        # 非对称损失（公式5：区分一致和不一致样本）
        y_argmax = torch.argmax(Y_tilde, dim=1)  # 伪标签
        p_argmax = torch.argmax(p, dim=1)  # 原始预测标签
        mask_same = (y_argmax == p_argmax)  # 一致样本掩码
        mask_diff = (y_argmax != p_argmax)  # 不一致样本掩码

        # 一致样本：交叉熵损失
        loss1 = F.cross_entropy(p[mask_same], y_argmax[mask_same], reduction='sum') if mask_same.any() else 0.0
        # 不一致样本：对称交叉熵损失
        loss2 = self.symmetric_cross_entropy(p[mask_diff], Y_star[self.num_classes:, :][mask_diff]).sum() if mask_diff.any() else 0.0

        # 总损失（公式6：加权求和）
        total_loss = (loss1 + self.beta * loss2) / (z.size()[0] + self.eps)

        # 反向传播更新模型
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 返回更新后的预测
        z_updated = self.extract_features(x)
        p_updated = self.classifier(z_updated)
        return p_updated

    def symmetric_cross_entropy(self, x, x_ema, alpha=0.5):
        """对称交叉熵损失（用于不一致样本）"""
        return -(1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - \
               alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
