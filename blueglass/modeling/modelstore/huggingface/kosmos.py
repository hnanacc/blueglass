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
from transformers import Kosmos2ForConditionalGeneration
from .base import HFModel


TOKEN_LIMIT = 1024


class Kosmos2(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.hf_id = ""
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(self.hf_id)
        self.procr = AutoProcessor.from_pretrained(self.hf_id)

        self.instruction_prompt = "<grounding> An image of"

    def forward(self, images: List[Tensor]) -> List[Dict]:
        batched_outputs = []
        for im in self.preprocess(images):
            inputs = self.procr(
                text=self.instruction_prompt, images=im, return_tensors="pt"
            )
            generation = self.model.generate(
                **inputs, max_new_tokens=TOKEN_LIMIT, use_cache=True, image_embeds=None
            )
            generated_text = self.procr.batch_decode(generation)
            text, entities = self.procr.post_process_generation(generated_text)
            batched_outputs.append((text, entities))
        return batched_outputs
