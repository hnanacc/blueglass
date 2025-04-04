# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from dataclasses import dataclass
import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Literal
from typing import NamedTuple, Dict, Any
from blueglass.configs import BLUEGLASSConf
from .autoencoder import AutoEncoder
from .utils.xformers import xformers_embedding_bag

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points))

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, decoder: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (decoder.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ decoder.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, decoder: Tensor):
    decoder_transpose = decoder.mT
    return TritonDecoder.apply(top_indices, top_acts, decoder_transpose)


# from elutherAI
def triton_decode_xformers(top_indices: Tensor, top_acts: Tensor, decoder: Tensor):
    return xformers_embedding_bag(top_indices, decoder, top_acts)


try:
    from .utils.kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        # decoder_impl = triton_decode
        decoder_impl = triton_decode_xformers


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class TopKFast(AutoEncoder):
    def __init__(self, conf: BLUEGLASSConf, feature_in_dim: int):
        super().__init__(conf, feature_in_dim)
        self.use_latents_bias = False
        self.use_feature_norm = False
        self.latents_topk_aux = conf.sae.latents_topk_aux
        self.coeff_topk_aux = conf.sae.loss_topk_aux_coeff

        self.device = DEVICE
        self.num_tokens_since_fired = torch.zeros(self.latents_dim)

        self.encoder = nn.Linear(
            self.feature_dim,
            self.latents_dim,
        )
        self.encoder.bias.data.zero_()

        self.decoder = nn.Parameter(self.encoder.weight.data.clone().contiguous())

        if self.use_feature_bias:
            self.feature_bias = nn.Parameter(torch.zeros(self.feature_dim))

        if self.conf.sae.use_decoder_norm:
            self.set_decoder_to_unit_norm(grads=False)

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.conf.sae.latents_topk, sorted=False))

    def decode_topkfast(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.decoder is not None, "Decoder weight was not initialized."

        dense = self.decoder
        y = decoder_impl(top_indices, top_acts, dense)
        if self.use_feature_bias:
            return y + self.feature_bias
        else:
            return y

    def encode(
        self, true_features: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if self.use_feature_bias:
            true_features = true_features - self.feature_bias

        out = self.encoder(true_features)
        interims = nn.functional.relu(out)

        # if self.use_latents_bias:
        #     interims = interims + self.latents_bias

        return interims, ctx

    def decode(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        top_values = ctx["raw_interims"]
        top_values = ctx["top_values"]
        top_indices = ctx["top_latents"]

        return self.decode_topkfast(top_values, top_indices), ctx

    def process_interim(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        ctx["raw_interims"] = interims
        topk_values, top_indices = self.select_topk(interims)
        ctx["top_values"] = topk_values
        ctx["top_latents"] = top_indices

        interims_topk = torch.zeros_like(interims).scatter(-1, top_indices, topk_values)

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

        dead_features = interims_dead_topk @ self.decoder[dead_latents_msk]

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
            "extra_min_dead_pct": self._min_dead_pct(),
            "extra_feature_seen_count": self.feature_seen_count,
        }

    @torch.no_grad()
    def set_decoder_to_unit_norm(self, grads=True):
        assert self.decoder is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.decoder.dtype).eps
        norm = torch.norm(self.decoder.data, dim=1, keepdim=True)
        self.decoder.data /= norm + eps
        if grads:
            self.remove_gradient_parallel_to_decoder_directions()

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder is not None, "Decoder weight was not initialized."
        assert self.decoder.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.grad,
            self.decoder.data,
            "d_sae feature_dim, d_sae feature_dim -> d_sae",
        )
        self.decoder.grad -= einops.einsum(
            parallel_component,
            self.decoder.data,
            "d_sae, d_sae feature_dim -> d_sae feature_dim",
        )
