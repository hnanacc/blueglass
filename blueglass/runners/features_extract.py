# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import torch
from blueglass.utils.logger_utils import setup_blueglass_logger
from collections import defaultdict
import pandas as pd
from blueglass.configs import BLUEGLASSConf
from blueglass.features import (
    FeatureInterceptor,
    FeatureStorage,
    Recorder,
    StandardRecorder,
)
from blueglass.runners import Runner
from blueglass.third_party.detectron2.utils import comm

from typing import List, Dict, TypeGuard, Any

logger = setup_blueglass_logger(__name__)


def is_list_of_dict(items: Any) -> TypeGuard[List[Dict[str, Any]]]:
    return isinstance(items, list) and all(isinstance(it, Dict) for it in items)


class FeatureExtractRunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.model = FeatureInterceptor(
            conf, self.build_model(conf), self.build_recorders(conf)
        )
        self.store = FeatureStorage(conf, conf.dataset.infer, conf.model.name)

    def build_recorders(self, conf: BLUEGLASSConf) -> Dict[str, Recorder]:
        return {name: StandardRecorder(name) for name in conf.feature.patterns}

    def run_step(self, batched_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        with torch.inference_mode():
            _, batched_features = self.model(batched_inputs, record=True)

        comm.synchronize()
        gathered_batched_features = comm.gather(batched_features)

        if not comm.is_main_process():
            return {}

        assert is_list_of_dict(
            gathered_batched_features
        ), "unexpected type from gather."

        merged_batched_features = defaultdict(list)

        for batched_features in gathered_batched_features:
            for fname, frame in batched_features.items():
                merged_batched_features[fname].append(frame)

        merged_batched_features = {
            pname: pd.concat(frames, ignore_index=True)
            for pname, frames in merged_batched_features.items()
        }

        self.store.write(merged_batched_features)

        return {}

    def train(self):
        raise NotImplementedError("unsupported. use infer.")

    def test(self):
        raise NotImplementedError("unsupported. use infer.")
