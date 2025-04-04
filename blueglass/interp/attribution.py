# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Dict, Any
import torch
from blueglass.configs import BLUEGLASSConf
from .base import Interpreter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetAttribution(Interpreter):
    def __init__(self, conf: BLUEGLASSConf, latents_dim: int, dataset_size: int):
        super().__init__(conf)
        self.top_props_id = []
        self.top_image_id = []
        self.top_props_per_latent = []
        self.top_image_per_latent = []

    def topk_image(self, latents_per_batch: Dict[str, Any]):
        # for each batch find topk latents.

        # compute topk latents for this batch.
        top_inds_cur_batch, top_vals_cur_batch = torch.topk(latents_per_image)

        # merge with the current topk latents.
        merged_inds_per_latent = torch.cat(
            self.top_inds_per_latent, top_latents_in_batch
        )

        # compute the new topk latents.
        self.top_inds_per_latent = torch.topk(merged_inds_per_latent)
        self.top_acts_per_latent = torch.topk(merged_acts_per_latent)

    def topk_proposals(self, batch: Dict[str, Any]):
        pass

    def _formatted(self, sample: Dict[str, Any]):
        pass

    def process(self, batched_inputs: Dict[str, Any], batched_outputs: Dict[str, Any]):
        latents = batched_outputs["latents"]
        img_ids = batched_inputs["image_ids"]

        # save the boxes of proposals instead.
        # save the

        pass
