# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import uuid
import gc
import wandb
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import Dict, Any, List, Tuple, Union
from functools import lru_cache
from fvcore.common.checkpoint import Checkpointer
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler
from blueglass.third_party.detectron2.utils import comm
from blueglass.third_party.detectron2.evaluation import (
    inference_on_dataset,
    DatasetEvaluator,
)
from blueglass.evaluation import build_evaluator
from blueglass.data.build import (
    build_test_dataloader,
    build_train_dataloader,
)
from blueglass.modeling.build import build_model
from blueglass.structures.types import is_comm_dict
from blueglass.configs import BLUEGLASSConf, FeaturePattern
from blueglass.third_party.detectron2.engine import create_ddp_model
from .utils import BestTracker

logger = setup_blueglass_logger(__name__)


class Runner:
    def __init__(self, conf: BLUEGLASSConf):
        self.conf = conf

        self.runner_name = str(self.conf.runner.name)
        self.runner_model_name = str(self.conf.model.name)
        self.model: nn.Module

        self.step = 1
        self.max_steps = conf.runner.max_steps
        self.warmup_steps = conf.runner.warmup_steps

        self.eval_period = conf.runner.eval_period
        self.logs_period = conf.runner.logs_period
        self.ckpt_period = conf.runner.ckpt_period
        self.precision = getattr(torch, conf.runner.precision)

        unique_id = uuid.uuid4().hex[:8]  # Short UUID (8 characters)
        self.conf.experiment.output_dir = f"{conf.experiment.output_dir}/{unique_id}/ckpts"
        
        assert (
            self.eval_period >= self.logs_period
        ), "invalid eval period and logs period, logs must be smaller than eval."

        assert (
            self.eval_period % self.logs_period == 0
        ), "invalid eval period and logs period, must be divisible."

    def build_scheduler(
        self,
        conf: BLUEGLASSConf,
        optimizer: Optimizer,
    ) -> LRScheduler:
        """
        Build a learning rate scheduler based on configuration.

        Args:
            conf: Configuration object containing scheduler settings
            optimizer: The optimizer to schedule

        Returns:
            LRScheduler: Configured learning rate scheduler

        Raises:
            ValueError: If an unsupported scheduler type is specified
        """
        if conf.runner.scheduler == "multistep":
            return optim.lr_scheduler.MultiStepLR(optimizer, conf.runner.milestones)
        if conf.runner.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, conf.runner.max_steps
            )

        raise ValueError("unsupported scheduler.")

    def build_optimizer(
        self,
        conf: BLUEGLASSConf,
        model: nn.Module,
    ) -> Optimizer:
        if conf.runner.optimizer == "adamw":
            return AdamW(
                model.parameters(),
                conf.runner.lr,
                conf.runner.betas,
                conf.runner.eps,
                conf.runner.weight_decay,
            )

        raise ValueError("unsupported optimizer.")

    def build_checkpointer(self, conf: BLUEGLASSConf, model, optimizer) -> Checkpointer:
        path = f"{conf.experiment.output_dir}/ckpts"
        os.makedirs(path, exist_ok=True)
        return Checkpointer(model, path, optimizer=optimizer)

    def build_infer_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        return build_test_dataloader(
            conf.dataset.infer, conf.dataset.batch_size, conf.num_data_workers
        )

    def build_test_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        return build_test_dataloader(
            conf.dataset.test, conf.dataset.batch_size, conf.num_data_workers
        )

    def build_train_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        return build_train_dataloader(
            conf.dataset.train, conf.dataset.batch_size, conf.num_data_workers
        )

    def build_evaluator(self, conf: BLUEGLASSConf) -> DatasetEvaluator:
        return build_evaluator(conf)

    def build_model(self, conf: BLUEGLASSConf) -> nn.Module:
        return create_ddp_model(build_model(conf))

    @lru_cache
    def prepare_filter_scheme(self, remove_io: bool = True) -> str:
        patterns = self.conf.feature.patterns.copy()
        if remove_io:
            patterns.remove(FeaturePattern.IO)
        patterns = "|".join(patterns) if len(patterns) > 0 else r"\w+"

        subpatns = self.conf.feature.sub_patterns
        subpatns = "|".join(subpatns) if len(subpatns) > 0 else r"\w+"

        layerids = self.conf.feature.layer_ids
        layerids = [str(li) for li in layerids]
        layerids = "|".join(layerids) if len(layerids) > 0 else r"\d+"

        return f"layer_({layerids}).({patterns}).({subpatns})"

    def process_records(
        self, gathered_records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        losses_dict = {}
        metric_dict = {}
        extras_dict = {}

        for rank, records in enumerate(gathered_records):
            raise NotImplementedError("Override in child class.")

        return losses_dict, metric_dict, extras_dict

    def register_metrics(self, records_dict: Dict[str, Any]):
        records_dict = {
            k: v.detach().cpu().item() if isinstance(v, Tensor) else v
            for k, v in records_dict.items()
        }

        gathered_records_dict = comm.gather(records_dict)

        if not comm.is_main_process():
            return

        assert is_comm_dict(
            gathered_records_dict
        ), "comm error! unexpected data format."

        extras_dict, losses_dict, metric_dict, visual_metric_dict = (
            self.process_records(gathered_records_dict)
        )

        assert (
            "losses_reduced" in losses_dict
        ), "expected losses in losses_dict at every publish step."

        if not np.isfinite(losses_dict["losses_reduced"]):
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration: {self.step}!\n"
                f"Losses: {losses_dict}"
            )

        if "metric_fitness" in metric_dict and self.best_tracker.is_best(
            metric_dict["metric_fitness"], self.step
        ):
            self.checkpoint()
        else:
            metric_dict["metric_fitness"] = self.best_tracker.best()

        lrs_dict = {
            f"lr/pg_{i}": group["lr"]
            for i, group in enumerate(self.optimizer.param_groups)
        }
        if self.conf.experiment.use_wandb:
            if len(visual_metric_dict) > 0:
                visual_metric_dict = {k: wandb.Image(v) for k, v in visual_metric_dict.items()}
            wandb.log(
                {
                    **losses_dict,
                    **lrs_dict,
                    **metric_dict,
                    **extras_dict,
                    **visual_metric_dict,
                },
                step=self.step,
            )

        logger.info(
            f"iter: {self.step:0>6}/{self.max_steps}. "
            f"metric_fitness: {metric_dict['metric_fitness']:.4f}. "
            f"losses_reduced: {losses_dict['losses_reduced']:.4f}. "
        )

    def _checkpoint(self):
        assert hasattr(self, "checkpointer"), "checkpointer not initialized."
        self.checkpointer.save(f"model_{self.step}")

    def run_step(self, batched_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("Override in child class.")

    def initialize_train_attrs(
        self,
    ) -> Tuple[DataLoader, nn.Module, Optimizer, LRScheduler, Checkpointer]:
        d = self.build_train_dataloader(self.conf)
        m = self.build_model(self.conf).train()
        o = self.build_optimizer(self.conf, m)
        s = self.build_scheduler(self.conf, o)
        c = self.build_checkpointer(self.conf, m, o)

        resume = self.conf.runner.resume
        if self.conf.runner.mode in {"test", "infer"}:
            c.resume_or_load(self.conf.model.checkpoint_path, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            ## TODO complete this part
            self.start_iter = self.iter + 1
            pass
        return d, m, o, s, c

    def initialize_test_attrs(self) -> Tuple[DataLoader, nn.Module, DatasetEvaluator]:
        d = self.build_test_dataloader(self.conf)
        e = self.build_evaluator(self.conf)
        m = (
            self.model.eval()
            if self.conf.runner.mode == "train"
            else self.build_model(self.conf)
        )
        return d, m, e

    def train(self) -> None:
        (
            self.dataloader,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointer,
        ) = self.initialize_train_attrs()
        self.best_tracker = BestTracker()
        self.grad_scaler = GradScaler("cuda")

        for self.step, data in zip(range(1, self.max_steps + 1), self.dataloader):
            records_dict = self.run_step(data)

            if self.step <= self.warmup_steps:
                continue

            records_dict["metrics"] = self.test()
            self.model = self.model.train()

            if self.step % self.logs_period == 0:
                self.register_metrics(records_dict)

            if self.step % self.ckpt_period == 0:
                self.checkpoint()

            del records_dict
            torch.cuda.empty_cache()
            gc.collect()

    def test(self) -> Dict[str, Any]:
        records_test_dict = {}
        if self.step % self.eval_period == 0 or self.conf.runner.mode == "test":
            dataloader, model, evaluator = self.initialize_test_attrs()
            records_test_dict = inference_on_dataset(model, dataloader, evaluator)
        return records_test_dict

    def infer(self) -> None:
        self.dataloader = self.build_infer_dataloader(self.conf)
        self.model.eval()
        for self.step, data in enumerate(self.dataloader):
            records_dict = self.run_step(data)

            del records_dict
            torch.cuda.empty_cache()
            gc.collect()

            if self.step % self.logs_period == 0:
                logger.info(f"Processed {self.step} / {len(self.dataloader)}")

    def checkpoint(self) -> None:
        assert hasattr(self, "checkpointer"), "checkpointer not initialized."
        # All processes must reach here before proceeding
        comm.synchronize()
        if not comm.is_main_process():
            return

        self._checkpoint()
        if self.conf.experiment.use_wandb:
            checkpoint_name = f"model_{self.step}"
            basename = "{}.pth".format(checkpoint_name)
            save_file = os.path.join(self.checkpointer.save_dir, basename)

            artifact = wandb.Artifact(
                name=f"{self.runner_model_name}-step-{self.step}",  # Unique name per checkpoint
                type=self.runner_name,
                description=f"Model checkpoint at step {self.step}",
                metadata={"step": self.step, "framework": "PyTorch"},
            )
            # Add the checkpoint file
            artifact.add_file(str(save_file))
            wandb.log_artifact(artifact)
        save_locally = self.conf.runner.save_ckpt_locally
        if save_locally:
            logger.info(
                "Checkpointing to storage locally is set to True, hence saving it locally."
            )
        else:
            os.remove(
                os.path.join(self.checkpointer.save_dir, f"model_{self.step}.pth")
            )
            logger.info(
                "Checkpointing to storage locally is set to False, hence deleting after saving it in wandb."
            )