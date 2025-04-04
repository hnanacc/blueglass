# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Dict, Any
import torch
from blueglass.configs import BLUEGLASSConf
from .base import Interpreter


class SAELatentDistribution(Interpreter):
    """
    SAE Latent Distributions.

    Compute various distributional analysis for latents of SAE.
    """

    def __init__(self, conf: BLUEGLASSConf, latents_dim: int, dataset_size: int):
        super().__init__(conf)

        self.latents_fire_actv_dist = torch.zeros(
            (latents_dim, dataset_size), device=self.device
        )
        self.latents_fire_freq_dist = torch.zeros((latents_dim,), device=self.device)
        self.mean_per_latent = torch.zeros((latents_dim,), device=self.device)
        self.spar_per_latent = torch.zeros((latents_dim,), device=self.device)

    def process(self, batched_inputs: Dict[str, Any], batched_outputs: Dict[str, Any]):
        pass
