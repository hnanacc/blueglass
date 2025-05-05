# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Dict, Any
import torch
from blueglass.configs import BLUEGLASSConf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Interpreter:
    def __init__(self, conf: BLUEGLASSConf):
        self.device = DEVICE
        self.conf = conf

    def process(self, batched_inputs: Dict[str, Any], batched_outputs: Dict[str, Any]):
        raise NotImplementedError("Override in child.")

    def interpret(self):
        raise NotImplementedError("Override in child.")
