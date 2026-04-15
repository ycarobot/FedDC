import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class Tokenizer(nn.Module):
    def __init__(self,
                 n_input_channels=3,
                 n_output_channels=128,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 max_pool=True,
                 activation=nn.ReLU,
                 n_conv_layers=1,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + [n_output_channels] * n_conv_layers
        self.conv_layers = nn.Sequential()
        for i in range(n_conv_layers):
            self.conv_layers.add_module(
                f'conv_{i}',
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=conv_bias)
            )
            self.conv_layers.add_module(f'norm_{i}', nn.BatchNorm2d(n_filter_list[i + 1]))
            if activation is not None:
                self.conv_layers.add_module(f'act_{i}', activation())
        if max_pool:
            self.conv_layers.add_module(
                'pool',
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride, padding=pooling_padding)
            )
        self.apply(self.init_weights)

    def sequence_length(self, n_channels=3, height=32, width=32):
        x = torch.randn(1, n_channels, height, width)
        x = self.conv_layers(x)
        return x.size(2) * x.size(3)

    def forward(self, x):
        return self.conv_layers(x).flatten(2).transpose(1, 2)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, sequence_length, positional_embedding='learnable'):
        super().__init__()
        if positional_embedding == 'learnable':
            self.pos_emb = nn.Parameter(torch.randn(1, sequence_length, embedding_dim))
        elif positional_embedding == 'sinusoidal':
            pos_emb = torch.zeros(sequence_length, embedding_dim)
            position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.pos_emb = nn.Parameter(pos_emb.unsqueeze(0), requires_grad=False)
        else:
            self.pos_emb = None

    def forward(self, x):
        if self.pos_emb is not None:
            x = x + self.pos_emb
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio=4.0, dropout=0., attention_dropout=0.1,
                 stochastic_depth=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embedding_dim * mlp_ratio), embedding_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = stochastic_depth

    def forward(self, x):
        # Stochastic Depth
        if self.training and torch.rand(1) < self.stochastic_depth:
            return x
        # Self-Attention
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        # MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, sequence_length, embedding_dim, seq_pool=True, dropout=0., attention_dropout=0.1,
                 stochastic_depth=0.1, num_layers=4, num_heads=2, mlp_ratio=1.0, num_classes=10,
                 positional_embedding='learnable'):
        super().__init__()
        self.seq_pool = seq_pool  # 核心修复：保留布尔值开关
        self.pos_emb = PositionalEmbedding(embedding_dim, sequence_length, positional_embedding)
        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, mlp_ratio, dropout, attention_dropout, stochastic_depth)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)
        # 核心修复：序列池化参数命名为seq_pool_weight，避免与布尔开关重名
        if seq_pool:
            self.seq_pool_weight = nn.Parameter(torch.randn(1, 1, embedding_dim))  # 重命名参数
            self.fc = nn.Linear(embedding_dim, num_classes)
        else:
            self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.pos_emb(x)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)

        # Sequence Pooling（核心修复：使用seq_pool_weight参数）
        if self.seq_pool:  # 布尔开关判断
            # 序列池化计算（CCT论文公式）
            attn_weights = F.softmax(torch.matmul(x, self.seq_pool_weight.transpose(1, 2)), dim=1)
            x = torch.matmul(attn_weights.transpose(1, 2), x).squeeze(1)
        else:
            x = x.mean(dim=1)  # 全局平均池化
        return self.fc(x)  # 修复：直接返回fc结果，避免外部重复分类