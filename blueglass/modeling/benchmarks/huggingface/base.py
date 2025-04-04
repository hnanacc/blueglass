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


TOKEN_LIMIT = 1024


class HFModel(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.procr = None

        self.instruction_prompt = [
            {"role": "user", "content": "Locate all objects in the image."}
        ]

        self.transform = T.Compose([T.ToPILImage(), T.Resize(800, max_size=1333)])

    def preprocess(self, images: List[Tensor]) -> List[Tensor]:
        return [self.transform(im) for im in images]

    def forward(self, batched_inputs: List[Dict]) -> List[Dict]:
        raise NotImplementedError("Please extend.")
