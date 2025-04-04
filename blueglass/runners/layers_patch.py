# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import torch
from blueglass.utils.logger_utils import setup_blueglass_logger
import numpy as np
from typing import List, Dict, TypeGuard, Any
from blueglass.configs import BLUEGLASSConf
from blueglass.features import FeatureInterceptor, FeatureStorage
from blueglass.runners import Runner
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from blueglass.third_party.detectron2.utils import comm


logger = setup_blueglass_logger(__name__)


def is_list_of_dict(items: Any) -> TypeGuard[List[Dict[str, Any]]]:
    return isinstance(items, list) and all(isinstance(it, Dict) for it in items)


class LayersPatchRunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.model = FeatureInterceptor(conf, self.build_model(conf), {})

    def process_records(
        self, gathered_records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        losses_dict, metrics_dict = defaultdict(list), {}

        for rank, records_per_rank in enumerate(gathered_records):
            for branch, records_per_branch in records_per_rank.items():
                if branch == "records_fwd":
                    for key, value in records_per_branch.items():
                        if "loss" in key:
                            losses_dict[f"losses/{key}"].append(float(value))

                if branch == "metrics":
                    assert rank == 0, "metrics received in rank>0."
                    assert "bbox" in records_per_branch, "bbox not in metrics."

                    for key, value in records_per_branch["bbox"].items():
                        metrics_dict[f"metrics/{key}"] = value

        reduced_losses_dict = {}
        for key, value in losses_dict.items():
            reduced_losses_dict[key] = sum(value)

        reduced_losses_dict["losses_reduced"] = reduced_losses_dict.pop("losses/loss")
        metrics_dict["metric_fitness"] = sum(
            [v for v in metrics_dict.values() if np.isfinite(v)]
        )

        return reduced_losses_dict, metrics_dict, {}

    @torch.inference_mode()
    def run_step(self, batched_inputs: List[Dict]):
        # with torch.no_grad():
        records_fwd = self.model(batched_inputs, patch=True)

        return {"records_fwd": records_fwd}

    def train(self):
        raise NotImplementedError("unsupported. use infer.")

    def infer(self):
        raise NotImplementedError("infer is not supported for this runner.")
