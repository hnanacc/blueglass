# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
import time
from copy import deepcopy
from PIL import Image
from typing import Dict, Any, List
import torch
from torch import nn
from blueglass.configs import BLUEGLASSConf


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKOFF_TIME_IN_SEC = 5
EXTENDED_WAIT_TIME_IN_SEC = 5 * 60

logger = setup_blueglass_logger(__name__)


class ClosedSourceModel(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__()
        self.device = DEVICE
        self.conf = conf
        self.reqs_since_last_wait = 0
        self.reqs_allowed_before_wait = 1500

    def preprocess(self, inputs: Dict[str, Any]):
        del inputs["image"]
        im = Image.open(inputs["file_name"])
        pm = deepcopy(self.prompt)
        return im, pm

    def forward(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        time.sleep(BACKOFF_TIME_IN_SEC)

        if self.reqs_since_last_wait >= self.reqs_allowed_before_wait:
            logger.info("wait threshold reached, on sleep.")
            time.sleep(EXTENDED_WAIT_TIME_IN_SEC)
            self.reqs_since_last_wait = 0

        self.reqs_since_last_wait += 1

        assert len(batched_inputs) == 1, "Closed source models do not support batch."
        assert not self.training, "Closed source models can't be trained."
        processed = [self.preprocess(bi) for bi in batched_inputs]
        responses = [self.send_request(im, pm) for im, pm in processed]
        return [self.postprocess(bi, rsp) for bi, rsp in zip(batched_inputs, responses)]

    def send_request(self, frame: Image.Image, prompt: Dict[str, Any]):
        raise NotImplementedError("Implement in the child class.")

    def postprocess(self, inputs: Dict[str, Any], response: str) -> Dict[str, Any]:
        raise NotImplementedError("Implement in the child class.")
