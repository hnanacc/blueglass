# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict, Any
import torch
from torch import nn, Tensor
from torchvision import transforms as T
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.image_utils import load_image
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from blueglass.configs import BLUEGLASSConf
from .base import HFModel


TOKEN_LIMIT = 1024


class QwenVL(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.hf_id = "Qwen/Qwen-VL"
        self.procr = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, trust_remote_code=True, device_map=self.device
        ).eval()

        self.instruction_prompt = "Generate the caption in English with grounding"

    def forward(self, images):
        batched_outputs = []
        for im in self.preprocess(images):
            inputs = {"image": im, "text": self.instruction_prompt}
            inputs = self.procr(**inputs, return_tensors="torch")
            generation = self.model.generate(**inputs.to(self.device))
            resp = self.procr.decode(generation, skip_special_tokens=False)
            batched_outputs.append(resp)
        return batched_outputs
