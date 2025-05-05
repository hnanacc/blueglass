# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
import wandb
import concurrent.futures
from omegaconf import OmegaConf
import numpy as np
import torch
import umap.umap_ as umap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from torch import Tensor, nn, autocast
from torch.optim import Optimizer, AdamW
from functools import lru_cache
from collections import defaultdict, ChainMap
from .utils import maybe_strip_ddp
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import Iterable, List, Dict, Any, Tuple, Union, Optional
from blueglass.configs import BLUEGLASSConf, Model, FeaturePattern, Precision
from blueglass.runners import Runner
from blueglass.modeling import build_model
from blueglass.modeling.saes import GroupedSAE
from blueglass.evaluation import SAEEvaluator
from blueglass.features import (
    FeatureDataset,
    FeatureInterceptor,
    Patcher,
    SAEPatcher,
    build_feature_dataloader,
)
from blueglass.data import build_test_dataloader
from blueglass.third_party.detectron2.evaluation import inference_on_dataset
from blueglass.third_party.detectron2.engine import create_ddp_model

logger = setup_blueglass_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
This runner assumes that we are training the SAEs with the same latent dimensions across layers/names/patterns/subpatterns
"""


class SAERunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.base_lr = conf.runner.lr
        self.patch_eval_period = conf.runner.patch_eval_period
        self.visuals_eval_period = conf.runner.visuals_eval_period
        self.warmup_steps = conf.runner.warmup_steps
        self.device = DEVICE
        self.feature_model = self._frozen(build_model(conf))
        self.max_grad_norm = conf.runner.max_grad_norm
        self.vanilla_fm_metrics = None

    def _frozen(self, model: nn.Module):
        for p in model.parameters():
            p.requires_grad = False
        return model.eval()

    def _prepare_model_for_store(self, conf) -> Union[Model, nn.Module]:
        return conf.model.name if conf.feature.use_cached else self.feature_model

    def build_train_dataloader(self, conf):
        return build_feature_dataloader(
            conf,
            conf.dataset.train,
            self._prepare_model_for_store(conf),
            "train",
            self.prepare_filter_scheme(),
            num_workers=self.conf.num_data_workers,
        )

    def build_model(self, conf) -> nn.Module:
        store_meta = FeatureDataset(
            conf,
            conf.dataset.train,
            self._prepare_model_for_store(conf),
            filter_scheme=self.prepare_filter_scheme(),
        ).infer_feature_meta()

        assert (
            "feature_dim_per_name" in store_meta
        ), "Feature dims not found in store meta."

        return create_ddp_model(
            GroupedSAE(conf, store_meta["feature_dim_per_name"]).to(self.device),
            broadcast_buffers=False,
        )

    def _compute_scaled_lr(self, conf: BLUEGLASSConf, model: nn.Module) -> float:
        module = (
            model.module
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model
        )
        return (
            conf.runner.lr
            if conf.runner.lr is not None
            else 1e-4 / ((module.latents_dim / 2**14) ** 0.5)
        )

    def build_optimizer(
        self,
        conf: BLUEGLASSConf,
        model: nn.Module,
    ) -> Optimizer:
        ## TODO: use params groups?
        assert conf.runner.optimizer == "adamw", "Unsupported optimizer for saes."
        return AdamW(
            model.parameters(),
            self._compute_scaled_lr(conf, model),
            conf.runner.betas,
            conf.runner.eps,
            conf.runner.weight_decay,
        )

    def run_step(self, batched_inputs: Dict[str, Any]):
        assert isinstance(self.model, GroupedSAE) or isinstance(
            self.model.module, GroupedSAE
        ), "Expected GroupedSAE, runner doesn't support other SAE types."
        assert self.model.training, "Model not in train mode."

        if self.step <= self.warmup_steps:
            _return = self.model(batched_inputs, branch="warmup")
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Warmup Step {self.step}/{self.warmup_steps}, LR: {current_lr:.4f}"
            )
            return _return

        with autocast("cuda", dtype=self.precision):
            records = self.model(batched_inputs, branch="autoenc")
            assert isinstance(records, Dict), "Unexpected from saes."

        self.optimizer.zero_grad()

        for name, item in records.items():
            if "loss_combined" in name:
                assert isinstance(item, Tensor), "Loss should be a tensor."
                item.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        model = maybe_strip_ddp(self.model)
        assert isinstance(model, GroupedSAE), "Expected module to be GroupedSAE"
        model.set_decoder_to_unit_norm()

        self.optimizer.step()
        self.scheduler.step()

        return records

    def _build_patchers(self) -> Dict[str, Patcher]:

        model = maybe_strip_ddp(self.model)

        built_patchers = {
            (t_name := model.transform_name(name, reverse=True)): SAEPatcher(
                t_name, sae
            )
            for name, sae in model.eval().sae_per_name.items()
        }
        return built_patchers

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

    def test(self) -> Dict[str, Any]:
        """
        running vanilla evaluation only once for the entire run
        """
        records_feature, records_patcher, records_visuals = {}, {}, {}
        if self.step % self.eval_period == 0:
            # Step 1. Measure SAE metrics on test data.
            fm = self._prepare_model_for_store(self.conf)
            fd = build_feature_dataloader(
                self.conf,
                self.conf.dataset.test,
                fm,
                "test",
                self.prepare_filter_scheme(),
            )
            fe = SAEEvaluator(self.conf, self.step)
            logger.info("Evaluation for SAE metrics.")
            records_feature = inference_on_dataset(self.model, fd, fe)

            # Step 2. Measure vanilla metrics on test data and feature model.
            if self.vanilla_fm_metrics is None:
                ds = build_test_dataloader(
                    self.conf.dataset.test, self.conf.dataset.batch_size
                )
                ev = self.build_evaluator(self.conf)
                logger.info("Evaluation for detection in VLM (vanilla).")
                vanilla_records_patcher = inference_on_dataset(
                    self.feature_model, ds, ev
                )

                self.vanilla_fm_metrics = vanilla_records_patcher
            else:
                vanilla_records_patcher = self.vanilla_fm_metrics

            for metric in vanilla_records_patcher.keys():
                for _metric_ in vanilla_records_patcher[metric].keys():
                    records_patcher[f"vanilla/{metric}_{_metric_}"] = (
                        vanilla_records_patcher[metric][_metric_]
                    )
        if self.step % self.patch_eval_period == 0:
            """
            Executes inference for all ad-hoc models registered with the base model
            """
            # Step 2a. Measure metrics with patcher on test data and feature model.
            test_patcher = self.patcher_test()
            records_patcher = records_patcher | test_patcher

        if self.step % self.visuals_eval_period == 0:
            records_visuals = self.visualise_metrics()

        return {**records_feature, **records_patcher, **records_visuals}

    def patcher_test(self) -> Dict[str, Any]:

        records_patcher = {}
        built_patchers = self._build_patchers()
        ds = build_test_dataloader(self.conf.dataset.test, self.conf.dataset.batch_size)
        for name, _single_patcher in built_patchers.items():
            single_patcher = {name: _single_patcher}
            dm = FeatureInterceptor(
                self.conf, self.feature_model, patchers_per_name=single_patcher
            )
            ev = self.build_evaluator(self.conf)
            logger.info(f"Evaluation for detection in VLM with patcher ({name}).")
            _records_patcher = inference_on_dataset(
                dm, ds, ev, fwd_kwargs={"patch": True}
            )
            for metric in _records_patcher.keys():
                for _metric_ in _records_patcher[metric].keys():
                    records_patcher[f"{name}/{metric}_{_metric_}"] = _records_patcher[
                        metric
                    ][_metric_]

        return records_patcher

    def plot_reduced_decoders(self, sae, direc="row") -> plt.Figure:
        decoder_weights = sae.sparse_codes.cpu().numpy()
        if direc == "column":
            decoder_weights = decoder_weights.T
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = reducer.fit_transform(decoder_weights)
        fig = plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))  # ✅ Create explicit Axes

        if umap_proj.shape[0] == 0 or np.isnan(umap_proj).any():
            logger.warning(" Warning: UMAP projection is empty or NaN. Skipping plot.")
            return fig

        ax.set_title(f"SAE Decoder using UMap ({direc})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.scatter(umap_proj[:, 0], umap_proj[:, 1], s=5, alpha=0.8)
        fig.tight_layout()

        return fig

    def visualize_decoder_weights(self, direc="row") -> dict:
        model = maybe_strip_ddp(self.model)
        records = {}

        # Helper to call plotter
        def plot_one(name, sae):
            fig = self.plot_reduced_decoders(sae, direc=direc)
            return (name, fig)

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for name, sae in model.eval().sae_per_name.items():
                futures.append(executor.submit(plot_one, name, sae))

            for future in concurrent.futures.as_completed(futures):
                name, fig = future.result()
                records[f"{name}/{direc}"] = fig

        return records

    def visualise_metrics(self) -> Dict[str, Any]:
        records_visuals = {}
        if self.step % self.conf.runner.visuals_eval_period == 0:
            visuals = {}
            _visuals = self.visualize_decoder_weights(direc="row")
            visuals = {**visuals, **_visuals}

            _visuals = self.visualize_decoder_weights(direc="column")
            visuals = {**visuals, **_visuals}

            visuals = {f"decoder_weights/{k}": v for k, v in visuals.items()}
            # TODO Fix visual plots
            if visuals is not None:
                for metric in visuals.keys():
                    records_visuals[f"visual_metrics/{metric}"] = visuals[metric]

            logger.info("Visual metrics have been updated.")

        return records_visuals
