# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from blueglass.configs import BLUEGLASSConf

from .base import HFModel


TOKEN_LIMIT = 1024


class DeepSeekVL(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.hf_id = ""
        self.procr = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.hf_id, trust_remote_code=True, device_map=self.device
        ).eval()

        self.instruction_prompt = "Locate all objects in the image."

    def forward(self, images: List[Tensor]):
        batched_outputs = []
        for im in self.preprocess(images):
            resp = self.model.chat(
                image=im,
                text=self.instruction_prompt,
                tokenizer=self.procr,
                do_sample=False,
            )
            batched_outputs.append(resp)
        return batched_outputs
