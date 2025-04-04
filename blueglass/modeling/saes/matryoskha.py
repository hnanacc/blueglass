# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import numpy as np
import torch
from torch import Tensor
from typing import Dict, Any, Tuple
from blueglass.configs import BLUEGLASSConf
from .batch_topk import BatchTopK


class Matryoshka(BatchTopK):
    def __init__(self, conf: BLUEGLASSConf, feature_dim: int):
        super().__init__(conf, feature_dim)
        self.group_sizes = [int(feature_dim * m) for m in conf.sae.group_multipliers]
        self.group_start = [0] + torch.cumsum(
            torch.tensor(self.group_sizes), dim=0
        ).tolist()
        self.group_count = len(self.group_sizes)
        self.latents_dim = sum(self.group_sizes)

        self._init_components()

    def decode(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        intermediate_pred_features = []

        pred_features = torch.zeros(self.feature_dim, device=self.device)

        for group_id in range(self.group_count):
            st_ind = self.group_start[group_id]
            en_ind = self.group_start[group_id + 1]

            decoder_slice = self.decoder.weight.T[st_ind:en_ind]
            interim_slice = interims[:, st_ind:en_ind]

            pred_features = pred_features + (interim_slice @ decoder_slice)

            intermediate_pred_features.append(pred_features)

        if self.use_feature_bias:
            pred_features = pred_features + self.feature_bias

        if self.use_feature_norm:
            pred_features = (
                pred_features * ctx["feature_norm_stdv"] + ctx["feature_norm_mean"]
            )

        ctx["intermediate_pred_features"] = intermediate_pred_features

        return pred_features, ctx

    def _loss_reconstr_bias(self, true_features: Tensor):
        return (
            self.coeff_reconstr
            * (self.feature_bias - true_features.float()).pow(2).mean()
        )

    def compute_losses(
        self,
        true_features: Tensor,
        pred_features: Tensor,
        interims: Tensor,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        loss_reconstr_per_group = [
            self._loss_reconstr(true_features, ipf)
            for ipf in ctx["intermediate_pred_features"]
        ]

        loss_rec_bias = self._loss_reconstr_bias(true_features)
        loss_sparsity = self._loss_sparsity(interims)
        loss_reconstr = (sum(loss_reconstr_per_group) + loss_rec_bias) / (
            len(loss_reconstr_per_group) + 1
        )
        loss_topk_aux = self._loss_topk_aux(true_features, pred_features, interims, ctx)

        return {
            "loss_combined": loss_reconstr + loss_sparsity + loss_topk_aux,
            "loss_reconstr": loss_reconstr,
            "loss_sparsity": loss_sparsity,
            "loss_topk_aux": loss_topk_aux,
            "loss_fts_bias": loss_rec_bias,
            "extra_norm_l0": self._norm_l0(interims),
            "extra_norm_l1": self._norm_l1(interims),
            "extra_dense_pct": self._dense_pct(),
            "extra_dead_pct": self._dead_pct(),
            "min_loss_reconstr": np.min(loss_reconstr_per_group),
            "max_loss_reconstr": np.max(loss_reconstr_per_group),
            "extra_feature_seen_count": self.feaure_seen_count,
            "extra_threshold_topk": self.threshold_interims,
        }
