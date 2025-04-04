# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict, Any
from blueglass.configs import BLUEGLASSConf
from .autoencoder import AutoEncoder


class Spline(AutoEncoder):
    """
    Spline Sparse Autoencoder (Experimental, Incomplete):

    Decompose features into their spline partitions so
    that they can be labelled as concepts. The partitions
    are selected sparsely which overcomes the problems of
    "white partitions" and "redundant partitions".

    See SplineCam: https://arxiv.org/abs/2302.12828.
    """

    def __init__(self, conf: BLUEGLASSConf, feature_in_dim: int):
        super().__init__(conf, feature_in_dim)
        self.latent_size = conf.sae.expansion_factor * feature_in_dim
        self.encoder = nn.Linear(self.feature_size, self.latent_size * 2, bias=False)

        self.partition_weights, self.partition_biases = self.compute_partitions()
        self.register_buffer("W_dec", self.partition_weights)
        self.register_buffer("b_dec", self.partition_biases)

    def compute_partitions(self):
        return torch.stack([]), torch.stack([])

    def encode(
        self, true_features: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        interims = self.encoder(true_features)

        return interims, ctx

    def process_interim(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        processed = torch.sigmoid(interims) * (interims > 0.5).float()
        return processed, ctx

    def decode(self, interims, ctx: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        scaled = (
            torch.matmul(interims, self.W_dec.transpose(1, 2)) * interims.unsqueeze(-1)
        ).sum(dim=1)
        offset = (interims.unsqueeze(-1) * self.b_dec.unsqueeze(0)).sum(dim=1)

        pred_features = scaled + offset

        return pred_features, ctx
