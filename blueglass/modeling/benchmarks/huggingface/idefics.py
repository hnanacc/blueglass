# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict
from torch import Tensor
from transformers import AutoProcessor, AutoModelForVision2Seq
from blueglass.configs import BLUEGLASSConf
from .base import HFModel


TOKEN_LIMIT = 1024


class IDEFICS2(HFModel):
    def __init__(self, conf: BLUEGLASSConf):

        super().__init__(conf)
        self.hf_id = "HuggingFaceM4/idefics2-8b"
        self.procr = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.hf_id)

        message_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Locate all the objects in the image."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text"},
                    {
                        "text": "Here the bounding box coordinates for all objects in the image."
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "And how about this image?"},
                ],
            },
        ]

        self.instruction_prompt = self.procr.apply_chat_template(
            message_template, add_generation_prompt=True
        )
        self.instruction_images = []

    def forward(self, images: List[Tensor]) -> List[Dict]:
        batched_outputs = []
        for image in self.preprocess(images):
            inputs = self.procr(
                text=self.instruction_prompt,
                images=self.instruction_images + [image],
                return_tensors="pt",
            ).to(self.device)
            generation = self.model.generate(**inputs, max_new_tokens=512)
            resp = self.procr.batch_decode(generation, skip_special_tokens=True)
            batched_outputs.append(resp)
        return batched_outputs
