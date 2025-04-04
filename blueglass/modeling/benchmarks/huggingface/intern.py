# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict, Any
import torch
from torch import Tensor
from torchvision import transforms as T
from transformers import AutoModel, AutoTokenizer
from blueglass.configs import BLUEGLASSConf

from .base import HFModel


TOKEN_LIMIT = 1024


class InternVL2(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.hf_id = "OpenGVLab/InternVL2-4B"
        self.procr = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.hf_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.instruction_prompt = "<image>\nPlease locate all the object in the image."

    def forward(self, images: List[Tensor]) -> List[Dict]:
        batched_outputs = []
        for im in self.preprocess(images):
            resp = self.model.chat(
                self.procr,
                im,
                self.instruction_prompt,
                max_new_tokens=TOKEN_LIMIT,
                do_sample=False,
            )
            batched_outputs.append({"pred_classes": resp, "boxes": resp})
        return batched_outputs
