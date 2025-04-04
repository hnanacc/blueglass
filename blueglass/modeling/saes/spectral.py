# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from torch import nn, Tensor, fft
from typing import List, Dict, Any
from blueglass.configs import BLUEGLASSConf
from .autoencoder import AutoEncoder


class Spectral(AutoEncoder):
    """
    Proposition:

    spectral SAE   <=> vector SAE
    spectral basis <=> vector basis (overcomplete)
    fourier basis  <=> standard basis (fixed, bad)
    """

    def __init__(self, conf: BLUEGLASSConf, feature_in_dim: int):
        super().__init__(conf, feature_in_dim)
        self.latent_size = conf.sae.expansion_factor * conf.sae.feature_size
        self.encoder = nn.Linear(self.feature_size, self.latent_size * 2, bias=False)
        self.decoder = nn.Linear(self.latent_size * 2, self.feature_size, bias=False)

    def init_parameters(self):
        # initialize decoder weights to standard fourier basis.
        freqs = torch.linspace(0, 1, self.decoder.weight.shape[0])
        self.decoder.weight.data = torch.sin(
            2
            * torch.pi
            * freqs[:, None]
            * torch.linspace(0, 1, self.decoder.weight.shape[1])
        )

    def preprocess(self, batched_inputs: List[Dict[str, Any]]) -> Tensor:
        return torch.tensor([1, 2])

    def encode(self, features: Tensor) -> Dict[str, Tensor]:
        # TODO: does fourier makes sense? probably not
        freqs = fft.rfft2(features, norm="ortho")
        return {"magnitudes": torch.abs(freqs), "phases": torch.angle(freqs)}

    def process_latents(self, latents: Dict[str, Any]) -> Dict[str, Any]:
        topk = torch.topk(latents["amplitudes"], self.topk).indices
        latents["magnitudes"][~topk] = 0.0
        latents["phases"][~topk] = 0.0
        latents["mask"] = topk
        return latents

    def decode(self, latents: Dict[str, Any]) -> Tensor:
        freqs = latents["magnitudes"] * torch.exp(1j * latents["phases"])
        return fft.irfft2(freqs, s=self.input_shape, norm="ortho")

    def forward(self, batched_inputs: List[Dict[str, Any]]):
        true_features = self.preprocess(batched_inputs)
        latents = self.encode(true_features)
        latents = self.process_latents(latents)
        pred_features = self.decode(latents)

        if self.training:
            return self.compute_loss(true_features, pred_features, latents["mask"])
        else:
            return self.postprocess(true_features, pred_features, latents)

    def compute_loss(
        self, true_featurs: Tensor, pred_featurs: Tensor, latent_mask: Tensor
    ):
        reconstr_loss = torch.mean((true_featurs - pred_featurs) ** 2)
        sparsity_loss = torch.mean(torch.abs(latent_mask))
        return {
            "loss_combined": reconstr_loss + self.sparse_coeff * sparsity_loss,
            "loss_reconstr": reconstr_loss,
            "loss_sparsity": sparsity_loss,
        }

    def postprocess(self, true_features, pred_features, latents) -> Dict[str, Any]:
        return {
            "true_features": true_features,
            "pred_features": pred_features,
            "latents": latents,
        }


import torch
import torch.nn as nn
import torch.fft


class FourierSparseAutoencoder(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__()
        self.topk = topk  # Number of top Fourier coefficients to retain
        self.encoder = nn.Sequential(
            torch.fft.FFT2(),  # 2D FFT for spatial data
            nn.Flatten(),
            nn.Linear(input_dim, latent_dim * 2),  # Real and imaginary parts
        )
        self.sparse_selector = TopKSelector(latent_dim * 2, k=sparsity_k)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, input_dim),
            torch.fft.IFFT2(),  # Inverse FFT
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Refine reconstruction
        )

    def forward(self, x):
        # Encoder
        fft_input = fft(x)
        x_fft = self.encoder(x)
        x_sparse = self.sparse_selector(x_fft)  # Select top-K coefficients
        # Decoder
        x_recon = self.decoder(x_sparse)
        return x_recon
