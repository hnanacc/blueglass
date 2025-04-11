# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Tuple, Any
from blueglass.configs import BLUEGLASSConf
from .autoencoder import AutoEncoder


class ReLU(AutoEncoder):
    def __init__(self, conf: BLUEGLASSConf, feature_in_dim: int):
        super().__init__(conf, feature_in_dim)
        self._init_components()

    def process_interim(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        interims = F.relu(interims)

        ctx["top_latents"] = (interims > self.threshold_interims).nonzero()

        return interims, ctx

    def compute_losses(
        self,
        true_features: Tensor,
        pred_features: Tensor,
        interims: Tensor,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes and returns all loss components and relevant auxiliary metrics.

        The returned dictionary includes:
        - Keys prefixed with "loss_" → shown in the "Losses" tab on Weights & Biases (wandb)
        - Keys prefixed with "extra_" → shown in the "Extras" tab on wandb

        Loss terms:
            - loss_reconstr: reconstruction loss between predicted and true features
            - loss_sparsity: regularization loss enforcing sparsity in the activation
            - loss_topk_aux: optional auxiliary term based on top-k statistics
            - loss_combined: sum of all three (total training objective)

        Extra metrics:
            - extra_norm_l0 / l1: L0 and L1 norms for activation sparsity
            - extra_dense_pct / dead_pct / min_dead_pct: density and dead neuron stats
            - extra_feature_seen_count: how many features have been seen (for bookkeeping)
        """

        loss_sparsity = self._loss_sparsity(interims)
        loss_reconstr = self._loss_reconstr(true_features, pred_features)

        return {
            "loss_combined": loss_reconstr + loss_sparsity,
            "loss_reconstr": loss_reconstr,
            "loss_sparsity": loss_sparsity,
            "extra_norm_l0": self._norm_l0(interims),
            "extra_norm_l1": self._norm_l1(interims),
            "extra_dense_pct": self._dense_pct(),
            "extra_dead_pct": self._dead_pct(),
            "extra_min_dead_pct": self._min_dead_pct(),
            "extra_feature_seen_count": self.feature_seen_count,
        }
