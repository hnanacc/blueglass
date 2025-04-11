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


class Phi3(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.hf_id = "microsoft/Phi-3-vision-128k-instruct"
        self.procr = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device,
        ).eval()

        messages = [
            {"role": "user", "content": "<image|>\nLocate all objects in the image."},
            {"role": "assistant", "content": "bboxes"},
            {"role": "user", "content": "<|image|>\nLocate all objects in the image"},
        ]
        self.instruction_images = []
        self.instruction_prompt = self.procr.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, images: List[Tensor]):
        batched_outputs = []
        for im in self.preprocess(images):
            inputs = self.procr(
                self.instruction_prompt,
                self.instruction_images + [im],
                return_tensors="pt",
            )
            generation = self.model.generate(
                **inputs,
                eos_token_id=self.procr.tokenizer.eos_token_id,
                max_token_length=TOKEN_LIMIT,
                temperature=0.0,
                do_sample=False,
            )
            generation = generation[:, inputs["input_ids"].shape[1] :]
            resp = self.procr.batch_decode(
                generation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            batched_outputs.append(resp)
        return batched_outputs
