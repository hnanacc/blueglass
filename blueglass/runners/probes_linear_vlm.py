# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0


from typing import Dict, Any, Tuple, List
import numpy as np
from torch import Tensor
from fvcore.common.checkpoint import Checkpointer
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from blueglass.utils.logger_utils import setup_blueglass_logger
from blueglass.runners.runner import Runner
from blueglass.features import FeatureInterceptor
from blueglass.configs.constants import FeaturePattern
from blueglass.modeling.probes import LinearProbedVLM
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from blueglass.configs import BLUEGLASSConf


logger = setup_blueglass_logger(__name__)


class VLMLinearProbeRunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.feature_pattern = FeaturePattern(conf.feature.patterns)
        self.infer_batch_size = conf.dataset.test_batch_size
        self.probe_fwd_period = conf.probe.fwd_period

    def build_model(self, conf: BLUEGLASSConf) -> nn.Module:
        return LinearProbedVLM(conf)

    def initialize_train_attrs(
        self,
    ) -> Tuple[DataLoader, nn.Module, Optimizer, LRScheduler, Checkpointer]:
        d = self.build_train_dataloader(self.conf)
        m = self.build_model(self.conf).train()
        o = self.build_optimizer(self.conf, m.probes.parameters())
        s = self.build_scheduler(self.conf, o)
        c = self.build_checkpointer(self.conf, m.probes, o)
        return d, m, o, s, c

    def initialize_test_attrs(self) -> Tuple[DataLoader, nn.Module, DatasetEvaluator]:
        d = self.build_test_dataloader(self.conf)
        m = self.model
        e = self.build_evaluator(self.conf)
        return d, m, e

    def process_records(
        self, gathered_records: List[Dict[str, Dict[str, Dict[str, Any]]]], metric_mode: str = "test"
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        losses_dict, metrics_dict, extras_dict, visual_metrics_dict = {}, {}, {}, {}

        for rank, records_per_rank in enumerate(gathered_records):
            for branch, records_per_branch in records_per_rank.items():
                if branch == "records_prb":
                    for layer_name, records_per_layer in records_per_branch.items():
                        for key, value in records_per_layer.items():
                            if key.startswith("extra"):
                                formatted = (
                                    f"extras/rank_{rank}.probe_{layer_name}.{key}"
                                )
                                extras_dict[formatted] = value

                            if key.startswith("loss"):
                                formatted = (
                                    f"losses/rank_{rank}.probe_{layer_name}.{key}"
                                )
                                losses_dict[formatted] = float(value)

                elif branch == "records_vlm":
                    pass

                elif branch == "metrics" and rank == 0:
                    for layer_name, metrics_per_layer in records_per_branch.items():
                        assert (
                            "bbox" in metrics_per_layer
                        ), f"expected bbox in metrics: {metrics_per_layer}"

                        for metric, value in metrics_per_layer["bbox"].items():
                            formatted = f"metrics/probe_{layer_name}.{metric}"
                            metrics_dict[formatted] = value

                else:
                    raise Exception(f"unsupported branch: {branch} in records.")

        if len(losses_dict) > 0:
            losses_dict["losses_reduced"] = sum(losses_dict.values())

        if len(metrics_dict) > 0:
            prefix = f"metrics_{metric_mode}_fitness"
            metrics_dict[prefix] = sum([v for v in metrics_dict.values() if np.isfinite(v)])

        return extras_dict, losses_dict, metrics_dict, visual_metrics_dict

    def run_step(self, batched_inputs: Dict[str, Any]) -> Dict[str, Any]:
        assert self.model.training, "model not in train mode for train."

        # records are stored per layer.
        records = {}

        # Step 1: Forward batch and record all features.
        records["records_vlm"] = self.model(batched_inputs, branch="extraction")
        assert isinstance(records["records_vlm"], Dict), "unexpected from vlm."

        if self.step % self.probe_fwd_period == 0:
            # Step 2: Forward probes with the collected features.
            records["records_prb"] = self.model(batched_inputs, branch="fwd_probes")
            assert isinstance(records["records_prb"], Dict), "unexpected from probe."

            # Step 3: Update parameters.
            self.optimizer.zero_grad()

            for records_per_layer in records["records_prb"].values():
                assert isinstance(records_per_layer, Dict), "unexpected out per layer."
                for loss_name, loss in records_per_layer.items():
                    assert isinstance(
                        loss_name, str
                    ), "unexpected loss name, not a str."
                    if loss_name.startswith("loss"):
                        assert isinstance(loss, Tensor), "received non-tensor loss."
                        loss.backward()

            self.optimizer.step()
            self.scheduler.step()

        return records
