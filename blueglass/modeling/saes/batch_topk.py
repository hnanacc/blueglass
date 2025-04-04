# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Any, Dict, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from blueglass.configs import BLUEGLASSConf
from .topk import TopK


class BatchTopK(TopK):
    def process_interim(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        interims = F.relu(interims)

        ctx["raw_interims"] = interims

        if self.training:
            interims_topk = torch.topk(
                interims.flatten(), self.latents_topk * interims.shape[0], dim=-1
            )

            ctx["top_latents"] = interims_topk.indices

            interims_topk = (
                torch.zeros_like(interims.flatten())
                .scatter(-1, interims_topk.indices, interims_topk.values)
                .reshape(interims.shape)
            )
            self._update_threshold(interims_topk)

        else:
            topk_latents_mask = interims > self.threshold_top_latents
            interims_topk = torch.where(
                topk_latents_mask,
                interims,
                torch.zeros_like(interims),
            )

            ctx["top_latents"] = topk_latents_mask.nonzero()

        return interims_topk, ctx
