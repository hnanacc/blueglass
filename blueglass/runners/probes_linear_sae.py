# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from .runner import Runner
from typing import List, Dict, Any
from blueglass.configs import BLUEGLASSConf


class SAELinearProbeRunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)

    def run_step(self, batched_inputs: List[Dict[str, Any]]):
        return []
