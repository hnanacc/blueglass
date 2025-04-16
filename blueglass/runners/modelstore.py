# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import numpy as np
import os
import wandb
import torch
from torch import Tensor, nn, autocast
from blueglass.runners.runner import Runner
from typing import Dict, Any, List, Tuple
from collections import defaultdict, ChainMap
from blueglass.configs import BLUEGLASSConf
from blueglass.modeling.build import build_model
from blueglass.third_party.detectron2.engine import create_ddp_model
from blueglass.utils.logger_utils import setup_blueglass_logger

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
        def strip_extra_prefix(key: str, substr: str) -> str:
            parts = key.rsplit("/", 1)
            if len(parts) == 2 and parts[1].startswith(substr):
                parts[1] = parts[1].removeprefix(substr)
            return "/".join(parts)

        losses_dict, metrics_dict, visual_metrics_dict, extras_dict = {}, {}, {}, {}

        reduced_records = defaultdict(list)

        for records_per_rank in gathered_records:
            for name, item in records_per_rank.items():
                reduced_records[name].append(item)

        reduced_records = {
            name: item if "metric" in name else np.mean(item)
            for name, item in reduced_records.items()
        }

        to_remove = []
        for name, item in reduced_records.items():
            if "loss" in name:
                losses_dict[name] = item
                continue

            if "extra" in name:
                extras_dict[name] = item
                continue

            if "metric" in name:
                metrics_data = dict(ChainMap(*item))

                # Split into visual and non-visual metrics

                visual_metrics = {
                    k: v for k, v in metrics_data.items() if "visual" in k
                }
                non_visual_metrics = {
                    f"metrics/{k}": v
                    for k, v in metrics_data.items()
                    if "visual" not in k
                }

                # Update target dicts
                visual_metrics_dict.update(visual_metrics)
                metrics_dict.update(non_visual_metrics)
                continue

        extras_dict = {
            f"extra/{strip_extra_prefix(k, 'extra_')}": v
            for k, v in extras_dict.items()
            if "visual" not in k
        }

        # Safely remove after iteration
        for key in to_remove:
            metrics_data.pop(key)

        if len(losses_dict) > 0:
            losses_dict["losses_reduced"] = sum(losses_dict.values())

        if len(metrics_dict) > 0:
            metrics_dict["metric_fitness"] = sum(metrics_dict.values())

            # Computing metric fitness based on each sae
            metric_fitness_dict = defaultdict(int)
            non_visual_metrics_data = {
                k: v for k, v in metrics_data.items() if "visual" not in k
            }
            for key, value in non_visual_metrics_data.items():

                if isinstance(key, str) and "/" in key:
                    prefix = key.split("/")[0]
                    metric_fitness_dict[f"metric_reduced/{prefix}"] += value

            metrics_dict = metrics_dict | metric_fitness_dict

        return extras_dict, losses_dict, metrics_dict, visual_metrics_dict

    def run_step(self, batched_inputs: Dict[str, Any]) -> Dict[str, Any]:

        if self.step <= self.warmup_steps:
            _return = self.model(batched_inputs)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Warmup Step {self.step}/{self.warmup_steps}, LR: {current_lr:.4f}"
            )
            return _return

        with autocast("cuda", dtype=self.precision):
            records = self.model(batched_inputs)

        self.optimizer.zero_grad()

        loss = records.pop("loss/loss")

        assert isinstance(loss, Tensor), "received non-tensor loss."

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.conf.runner.max_grad_norm
        )
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()

        return records

    def infer(self):
        raise NotImplementedError("infer is not supported for this runner.")
