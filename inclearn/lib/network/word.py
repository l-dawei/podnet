import logging
import os
import pickle
from mindspore import nn
import mindspore as ms
import mindspore.ops as ops
# import numpy as np
# import torch
from scipy.io import loadmat
# from torch import nn
# from torch.nn import functional as F

import gensim
from inclearn.lib.data import fetch_word_embeddings

from .mlp import MLP

logger = logging.getLogger(__name__)


class Word2vec(nn.Cell):

    def __init__(
        self,
        embeddings="googlenews",
        dataset="cifar100",
        mlp_dims=None,
        use_bn=True,
        input_dropout=0.2,
        hidden_dropout=0.5,
        device=None,
        noise_dimension=50,
        noise_type="normal",
        freeze_embedding=True,
        scale_embedding=None,
        data_path=None
    ):
        super().__init__()

        self.emb, _ = get_embeddings(dataset, embeddings, frozen=freeze_embedding, path=data_path)
        l2_normalize = ops.L2Normalize(axis=-1)
        if isinstance(scale_embedding, list):
            logger.info(f"Scaling semantic embedding in {scale_embedding}.")
            self.emb.weight.data = Scaler(scale_embedding).fit_transform(self.emb.weight.data)
        elif isinstance(scale_embedding, str) and scale_embedding == "l2":
            self.emb.weight.data = l2_normalize(self.emb.weight.data)

        semantic_dim = self.emb.weight.shape[1]
        logger.info(f"Semantic dimension: {semantic_dim}.")

        if mlp_dims is not None:
            self.mlp = MLP(
                input_dim=noise_dimension + semantic_dim,
                hidden_dims=mlp_dims,
                use_bn=use_bn,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout
            )
        else:
            self.mlp = None

        self.noise_dimension = noise_dimension
        self.noise_type = noise_type
        self.to(device)
        self.device = device
        self.out_dim = mlp_dims[-1]

        self.linear_transform = None

    def add_linear_transform(self, bias=False):
        self.linear_transform = nn.Linear(self.out_dim, self.out_dim, bias=bias)
        #self.linear_transform.weight.data = torch.eye(self.out_dim)
        #self.linear_transform.weight.data += torch.empty(self.out_dim, self.out_dim).normal_(mean=0, std=0.1)
        if bias:
            self.linear_transform.bias.data.fill_(0.)
        self.linear_transform.to(self.device)

    def forward(self, x, only_word=False):
        word = self.emb(x)

        if only_word:
            return word

        if self.noise_dimension:
            if self.noise_type == "normal":
                noise = torch.randn(len(x), self.noise_dimension).to(word.device)
            elif self.noise_type == "uniform":
                noise = torch.rand(len(x), self.noise_dimension).to(word.device)
            else:
                raise ValueError(f"Unknown noise type {self.noise_type}.")

        if self.mlp:
            fake_features = self.mlp(torch.cat((word, noise), dim=-1))
            if self.linear_transform:
                fake_features = self.linear_transform(fake_features)
            return fake_features

        return word




class Scaler:
    """
    Transforms each channel to the range [a, b].
    """

    def __init__(self, feature_range):
        self.feature_range = feature_range

    def fit(self, tensor):
        data_min = torch.min(tensor, dim=0)[0]
        data_max = torch.max(tensor, dim=0)[0]
        data_range = data_max - data_min

        # Handle null values
        data_range[data_range == 0.] = 1.

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        return self

    def transform(self, tensor):
        return tensor.mul_(self.scale_).add_(self.min_)

    def inverse_transform(self, tensor):
        return tensor.sub_(self.min_).div_(self.scale_)

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)
