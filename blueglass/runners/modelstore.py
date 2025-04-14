# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
import numpy as np
import torch
from torch import Tensor, nn
from blueglass.runners.runner import Runner
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from blueglass.configs import BLUEGLASSConf
from blueglass.modeling.build import build_model
from blueglass.third_party.detectron2.engine import create_ddp_model

logger = setup_blueglass_logger(__name__)


class ModelstoreRunner(Runner):

    def build_model(self, conf: BLUEGLASSConf) -> nn.Module:
        model = build_model(conf)
        """
        Freeze the model according to your need
        """
        model = create_ddp_model(model)
        return model

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

    def run_step(self, batched_inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.optimizer.zero_grad()

        with torch.autocast("cuda", torch.float16):
            records_fwd = self.model(batched_inputs)

        losses = records_fwd["loss"]
        assert isinstance(losses, Tensor), "received non-tensor loss."

        self.grad_scaler.scale(losses).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.runner.max_grad_norm)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()

        return {"records_fwd": records_fwd}

    def infer(self):
        raise NotImplementedError("infer is not supported for this runner.")
