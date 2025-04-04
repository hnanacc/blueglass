# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from safetensors.torch import load_model
import torch
from torch import nn
from typing import Dict, Any
from blueglass.configs import BLUEGLASSConf, SAEVariant
from .autoencoder import AutoEncoder
from .relu import ReLU
from .topk import TopK
from .topk_fast import TopKFast
from .matryoskha import Matryoshka
from .spectral import Spectral
from .spline import Spline
from .crosscoder import Crosscoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GroupedSAE(nn.Module):
    def __init__(self, conf: BLUEGLASSConf, feature_dim_per_name: Dict[str, int]):
        super().__init__()
        self.conf = conf
        self.device = DEVICE
        self.sae_per_name = nn.ModuleDict(
            {
                self.transform_name(name): build_sae(conf, feature_dim)
                for name, feature_dim in feature_dim_per_name.items()
            }
        )
        self.latents_dim = self._infer_latents_dim()

    def __len__(self):
        return len(self.sae_per_name)

    def _infer_latents_dim(self) -> int:
        latents_per_name = set([sae.latents_dim for sae in self.sae_per_name.values()])
        assert len(latents_per_name) == 1, "Expected all latents dims to be same."
        return latents_per_name.pop()

    def transform_name(self, name: str, reverse: bool = False) -> str:
        if reverse:
            return name.replace("__", ".")
        else:
            return name.replace(".", "__")

    def forward(
        self, batched_inputs_per_name: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, Any]:
        records_per_name = {
            name: self.sae_per_name[self.transform_name(name)](
                batched_inputs, *args, **kwargs
            )
            for name, batched_inputs in batched_inputs_per_name.items()
        }

        return {
            f"{name}/{attr}": items
            for name, records in records_per_name.items()
            for attr, items in records.items()
        }

    @torch.no_grad()
    def set_decoder_to_unit_norm(self, grads=True):
        for sae in self.sae_per_name.values():
            assert isinstance(sae, AutoEncoder), "Invalid SAE class."
            sae.set_decoder_to_unit_norm(grads)


def build_sae(conf: BLUEGLASSConf, feature_in_dim: int) -> AutoEncoder:
    match conf.sae.variant:
        case SAEVariant.RELU:
            return ReLU(conf, feature_in_dim)
        case SAEVariant.TOPK:
            return TopK(conf, feature_in_dim)
        case SAEVariant.TOPK_FAST:
            return TopKFast(conf, feature_in_dim)
        case SAEVariant.MATRYOSHKA:
            return Matryoshka(conf, feature_in_dim)
        case SAEVariant.SPECTRAL:
            return Spectral(conf, feature_in_dim)
        case SAEVariant.SPLINE:
            return Spline(conf, feature_in_dim)
        case SAEVariant.CROSSCODER:
            return Crosscoder(conf, feature_in_dim)
        case unsupported:
            raise NotImplementedError(f"Unsupported SAE variant: {unsupported}.")
