# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from blueglass.configs import BLUEGLASSConf
from .autoencoder import AutoEncoder


class TopK(AutoEncoder):
    def __init__(self, conf: BLUEGLASSConf, feature_in_dim: int):
        super().__init__(conf, feature_in_dim)

        self.latents_topk = conf.sae.latents_topk
        self.latents_topk_aux = conf.sae.latents_topk_aux
        self.coeff_topk_aux = conf.sae.loss_topk_aux_coeff

        self._init_components()

    def process_interim(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        interims = F.relu(interims)
        ctx["raw_interims"] = interims

        interims_topk = torch.topk(interims, self.latents_topk, dim=-1)

        ctx["top_latents"] = interims_topk.indices

        interims_topk = torch.zeros_like(interims).scatter(
            -1, interims_topk.indices, interims_topk.values
        )

        return interims_topk, ctx

    def _loss_topk_aux(
        self,
        true_features: Tensor,
        pred_features: Tensor,
        interims: Tensor,
        ctx: Dict[str, Any],
    ):
        assert "raw_interims" in ctx, "Expected raw_interims in ctx."
        dead_latents_msk = self.latents_dead_since >= self.threshold_dead

        if not dead_latents_msk.any():
            return torch.tensor(0.0, device=self.device)

        interims_dead_topk = torch.topk(
            ctx["raw_interims"][:, dead_latents_msk],
            min(self.latents_topk_aux, int(dead_latents_msk.sum())),
            dim=-1,
        )

        interims_dead_topk = torch.zeros_like(interims[:, dead_latents_msk]).scatter(
            -1, interims_dead_topk.indices, interims_dead_topk.values
        )

        dead_features = interims_dead_topk @ self.decoder.weight.T[dead_latents_msk]

        live_error = true_features.float() - pred_features.float()
        dead_error = true_features.float() - dead_features.float()

        return self.coeff_topk_aux * (live_error - dead_error).pow(2).mean().nan_to_num(
            0.0
        )

    def compute_losses(
        self,
        true_features: Tensor,
        pred_features: Tensor,
        interims: Tensor,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        loss_sparsity = self._loss_sparsity(interims)
        loss_reconstr = self._loss_reconstr(true_features, pred_features)
        loss_topk_aux = self._loss_topk_aux(true_features, pred_features, interims, ctx)

        return {
            "loss_combined": loss_reconstr + loss_sparsity + loss_topk_aux,
            "loss_reconstr": loss_reconstr,
            "loss_sparsity": loss_sparsity,
            "loss_topk_aux": loss_topk_aux,
            "extra_norm_l0": self._norm_l0(interims),
            "extra_norm_l1": self._norm_l1(interims),
            "extra_dense_pct": self._dense_pct(),
            "extra_dead_pct": self._dead_pct(),
            "extra_feature_seen_count": self.feature_seen_count,
        }
