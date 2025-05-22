# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Any, Tuple, Literal, Optional, Callable, Union
from blueglass.configs import BLUEGLASSConf
from blueglass.third_party.detectron2.utils import comm
from mmdet.utils import AvoidCUDAOOM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROW_CHUNK_SIZE = 10000


class TransCoder(nn.Module):
    """AutoEncoders

    ::Base class for Sparse AutoEncoders.

    Provides methods and common utilities.
    """

    latents_dead_since: Tensor
    latents_fire_count: Tensor
    feature_seen_count: Tensor
    threshold_interims: Tensor

    def __init__(self, conf: BLUEGLASSConf, feature_dim: int):
        super().__init__()
        self.device = DEVICE
        self.conf = conf

        self.use_feature_norm = conf.sae.use_feature_norm
        self.use_feature_bias = conf.sae.use_feature_bias
        self.use_latents_bias = conf.sae.use_latents_bias
        self.use_decoder_norm = conf.sae.use_decoder_norm

        self.feature_dim = feature_dim
        self.latents_dim = conf.sae.expansion_factor * feature_dim

        self.threshold_dead = conf.sae.threshold_latents_dead
        self.min_threshold_dead = conf.sae.min_threshold_latents_dead
        self.threshold_dense = conf.sae.threshold_latents_dense
        self.threshold_urate = conf.sae.threshold_update_rate

        self.coeff_reconstr = conf.sae.loss_reconstr_coeff
        self.coeff_sparsity = conf.sae.loss_sparsity_coeff

        self.register_buffer("latents_dead_since", torch.zeros(self.latents_dim))
        self.register_buffer("latents_fire_count", torch.zeros(self.latents_dim))
        self.register_buffer("feature_seen_count", torch.tensor(0, dtype=torch.int))
        self.register_buffer(
            "threshold_interims", torch.tensor(conf.sae.threshold_top_latents)
        )

    def _init_components(self):
        self.encoder = nn.Linear(self.feature_dim, self.latents_dim, bias=False)
        self.decoder = nn.Linear(self.latents_dim, self.feature_dim, bias=False)

        self.decoder.weight.data[:] = self.encoder.weight.t().data
        self.set_decoder_to_unit_norm(grads=False)

        if self.use_feature_bias:
            self.feature_bias = nn.Parameter(torch.zeros(self.feature_dim))
        if self.use_latents_bias:
            self.latents_bias = nn.Parameter(torch.zeros(self.latents_dim))

    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        branch: Literal["warmup", "autoenc"] = "autoenc",
    ):
        match branch:
            case "warmup":
                return self.forward_warmup(batched_inputs)
            case "autoenc":
                return self.forward_autoenc(batched_inputs)
            case unsupported:
                raise ValueError(f"Unsupported branch: {unsupported}")

    def forward_warmup(self, batched_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.use_feature_bias:
            return {}

        features, _ = self.preprocess(batched_inputs, {})
        features_sum = features.sum(dim=0)
        features_cnt = torch.tensor(
            features.shape[0], dtype=torch.int, device=self.device
        )

        comm.all_reduce(features_sum)
        comm.all_reduce(features_cnt)

        assert features_sum.shape[0] == features.shape[-1], "Unexpected features sum."
        assert features_cnt > 0, "Unexpected features count."

        # initialize the feature bias as mean of features.
        self.feature_bias.data = features_sum / features_cnt

        return {}

    def forward_autoenc(self, batched_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:

        true_input_features, ctx = self.preprocess(batched_inputs[0], {})
        true_output_features, ctx = self.preprocess(batched_inputs[1], {})
        prep_interims, ctx = self.encode(true_input_features, ctx)
        posp_interims, ctx = self.process_interim(prep_interims, ctx)
        pred_features, ctx = self.decode(posp_interims, ctx)

        if self.training:
            self.update_features_metrics(true_input_features, pred_features)
            self.update_interims_metrics(posp_interims)
            return self.compute_losses(
                true_output_features, pred_features, posp_interims, ctx
            )
        else:
            return self.postprocess(
                batched_inputs, true_output_features, pred_features, posp_interims, ctx
            )

    def _norm(
        self, features: Tensor, ctx: Dict[str, Any], eps: float = 1e-5
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        mean = features.mean(dim=-1, keepdim=True)
        features = features - mean

        stdv = features.std(dim=-1, keepdim=True)
        features = features / (stdv + eps)

        ctx["feature_norm_mean"] = mean
        ctx["feature_norm_stdv"] = stdv

        return features, ctx

    def _denorm(
        self, features: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        assert "feature_norm_mean" in ctx, "Expected feature_norm_mean in ctx."
        assert "feature_norm_stdv" in ctx, "Expected feature_norm_stdv in ctx."
        return (features * ctx["feature_norm_stdv"]) + ctx["feature_norm_mean"], ctx

    def _update_threshold(self, interims: Tensor):
        pos_mask = interims > 0

        if not pos_mask.any():
            return

        min_positive = interims[pos_mask].min()
        self.threshold_interims = (
            1 - self.threshold_urate
        ) * self.threshold_interims + self.threshold_urate * min_positive

    def preprocess(
        self, batched_inputs: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        features = batched_inputs["features"].to(self.device)
        assert isinstance(features, Tensor), "Expected features to be Tensors."
        return features, ctx

    def encode(
        self, true_features: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if self.use_feature_norm:
            true_features, ctx = self._norm(true_features, ctx)

        if self.use_feature_bias:
            true_features = true_features - self.feature_bias

        interims = self.encoder(true_features)

        if self.use_latents_bias:
            interims = interims + self.latents_bias

        return interims, ctx

    def process_interim(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError("Override in child class.")

    def decode(
        self, interims: Tensor, ctx: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        pred_features = self.decoder(interims)

        if self.use_feature_bias:
            pred_features = pred_features + self.feature_bias

        if self.use_feature_norm:
            pred_features, ctx = self._denorm(pred_features, ctx)

        return pred_features, ctx

    def _loss_reconstr(self, true_features: Tensor, pred_features: Tensor) -> Tensor:
        return (
            self.coeff_reconstr
            * (pred_features.float() - true_features.float()).pow(2).mean()
        )

    def _loss_sparsity(self, interims: Tensor) -> Tensor:
        return self.coeff_sparsity * self._norm_l1(interims)

    @AvoidCUDAOOM.retry_if_cuda_oom
    def _norm_l0(self, interims: Tensor) -> Tensor:
        return (interims > 0).float().sum(dim=-1).mean()

    @AvoidCUDAOOM.retry_if_cuda_oom
    def _norm_l1(self, interims: Tensor) -> Tensor:
        return interims.float().abs().sum(dim=-1).mean()

    @AvoidCUDAOOM.retry_if_cuda_oom
    def _dense_pct(self) -> Tensor:
        return (
            self.latents_fire_count > self.threshold_dense
        ).sum() / self.feature_seen_count.float()

    @AvoidCUDAOOM.retry_if_cuda_oom
    def _dead_pct(self) -> Tensor:
        num_dead = (self.latents_dead_since > self.threshold_dead).sum()
        return (num_dead / float(self.latents_dim)) * 100

    @AvoidCUDAOOM.retry_if_cuda_oom
    def _min_dead_pct(self) -> Tensor:
        num_dead = (self.latents_dead_since > self.min_threshold_dead).sum()
        return (num_dead / float(self.latents_dim)) * 100

    @torch.no_grad()
    def update_features_metrics(self, true_features: Tensor, _: Tensor):
        cur_feature_seen_cnt = torch.tensor(
            true_features.shape[0], dtype=torch.int, device=self.device
        )
        cur_feature_seen_cnt = cur_feature_seen_cnt.to(self.device)
        comm.all_reduce(cur_feature_seen_cnt)
        self.feature_seen_count += cur_feature_seen_cnt

    @torch.no_grad()
    def update_interims_metrics(self, interims: Tensor):
        cur_latents_fire_cnt = (interims > 0).sum(dim=0)
        comm.all_reduce(cur_latents_fire_cnt)
        self.latents_fire_count += cur_latents_fire_cnt

        cur_feature_seen_cnt = torch.tensor(
            interims.shape[0], dtype=torch.int, device=self.device
        )
        cur_feature_seen_cnt = cur_feature_seen_cnt.to(self.device)
        comm.all_reduce(cur_feature_seen_cnt)

        self.latents_dead_since += cur_feature_seen_cnt
        self.latents_dead_since[(cur_latents_fire_cnt > 0)] = 0

    @torch.no_grad()
    def set_decoder_to_unit_norm(self, grads=True):
        if not self.use_decoder_norm:
            return

        normed = self.decoder.weight / self.decoder.weight.norm(dim=0, keepdim=True)
        self.decoder.weight.data = normed

        if not grads:
            return

        assert (
            self.decoder.weight.grad is not None
        ), "Gradient norm should be used in train and after backward."

        self.decoder.weight.grad -= (self.decoder.weight.grad * normed).sum(
            dim=0, keepdim=True
        ) * normed

    def compute_losses(
        self,
        true_features: Tensor,
        pred_features: Tensor,
        interims: Tensor,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError("Override in child class.")

    def postprocess(
        self,
        batched_inputs: Dict[str, Any],
        true_features: Tensor,
        pred_features: Tensor,
        interims: Tensor,
        ctx: Dict[str, Any],
    ):
        assert "top_latents" in ctx, "Expected top_latents in ctx."
        return {
            **self.compute_losses(true_features, pred_features, interims, ctx),
            "pred_features": pred_features,
            "prep_interims": ctx["raw_interims"],
            "posp_interims": interims,
            "latents_dead_since": self.latents_dead_since,
            "latents_fire_count": self.latents_fire_count,
            "top_latents": ctx["top_latents"],
        }

    @property
    def sparse_codes(self) -> Tensor:
        decoder = self.decoder
        return decoder.weight.T.detach()  # [N, D]

    @torch.no_grad()
    def set_knockoff_columns(self, column_indices, feature_bias=False):
        self.decoder.weight[:, column_indices] = 0
        if feature_bias:
            self.feature_bias[column_indices] = 0
