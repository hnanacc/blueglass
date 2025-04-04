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
        loss_sparsity = self._loss_sparsity(interims)
        loss_reconstr = self._loss_reconstr(true_features, pred_features)

        return {
            "loss_combined": loss_reconstr + loss_sparsity,
            "loss_reconstr": loss_reconstr,
            "loss_sparsity": loss_sparsity,
            "norm_l0": self._norm_l0(interims),
            "norm_l1": self._norm_l1(interims),
            "dense_pct": self._dense_pct(),
            "dead_pct": self._dead_pct(),
            "feature_seen_count": self.feaure_seen_count,
        }
