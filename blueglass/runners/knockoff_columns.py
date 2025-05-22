# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import gc
import copy
import wandb
from blueglass.utils.logger_utils import setup_blueglass_logger
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
from functools import lru_cache
from fvcore.common.checkpoint import Checkpointer
from collections import defaultdict, ChainMap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import umap.umap_ as umap
from sklearn.manifold import TSNE
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler

from .utils import maybe_strip_ddp
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
from blueglass.configs.utils import load_blueglass_from_wandb
from blueglass.modeling.saes import GroupedSAE
from blueglass.evaluation import SAEEvaluator
from blueglass.runners.saes import SAERunner
from blueglass.features import (
    FeatureDataset,
    FeatureInterceptor,
    Patcher,
    SAEPatcher,
    build_feature_dataloader,
)

from .utils import BestTracker

logger = setup_blueglass_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KnockoffColumns(SAERunner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.conf = conf

        self.runner_name = str(self.conf.runner.name)
        self.runner_model_name = str(self.conf.model.name)

        self.infer_step = 1
        self.max_steps = conf.runner.max_steps
        self.warmup_steps = conf.runner.warmup_steps

        self.eval_period = conf.runner.eval_period
        self.eval_knockoff_period = conf.runner.eval_knockoff_period
        self.logs_period = conf.runner.logs_period
        self.precision = getattr(torch, conf.runner.precision)

        self.device = DEVICE
        self.feature_model = self._frozen(build_model(conf))
        self.feature_model = self.feature_model
        self.vanilla_metrics = None
        self.column_rank_indices = None

        assert isinstance(self.precision, torch.dtype), "Invalid precision."

        assert (
            self.eval_period >= self.logs_period
        ), "invalid eval period and logs period, logs must be smaller than eval."

        assert (
            self.eval_period % self.logs_period == 0
        ), "invalid eval period and logs period, must be divisible."

    def build_infer_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        return build_test_dataloader(
            conf.dataset.infer, conf.dataset.test_batch_size, conf.num_data_workers
        )

    def prepare_metadata(self, conf: BLUEGLASSConf) -> Dict[str, Any]:
        return FeatureDataset(
            conf,
            conf.dataset.infer,
            conf.model.name,
            filter_scheme=self.prepare_filter_scheme(self.sae_conf),
        ).infer_feature_meta()

    def build_saes_model(self) -> nn.Module:
        """ "
        Loads the SAEs from the different checkpoints and creates a model for infer/test mode while overriding the
        blueglass config with the wandb config.
        The wandb config is used to load the correct model and the correct feature patterns.
        """

        filters = ["decoder_mlp"]
        filters = []
        patterns = self.sae_conf.feature.patterns

        self.sae_conf.feature.patterns = (
            [p for p in patterns if any(f in p.value for f in filters)]
            if filters
            else patterns
        )

        metadata = self.prepare_metadata(self.sae_conf)

        assert (
            "feature_dim_per_name" in metadata
        ), "Feature dims not found in store meta."

        m = GroupedSAE(self.sae_conf, metadata["feature_dim_per_name"]).to(self.device)
        assert self.conf.sae.checkpoint_path is not None, "Require SAE checkpoint."
        ckpt = torch.load(
            self.conf.sae.checkpoint_path, map_location="cpu", weights_only=False
        )
        missing_keys, unexpected_keys = m.load_state_dict(ckpt["model"], strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logger.warning(
                f"Missing keys in state_dict: {missing_keys}. "
                f"Unexpected keys in state_dict: {unexpected_keys}."
            )
        return m

    def build_model(self, conf) -> nn.Module:

        model = self.build_saes_model()
        return model.eval().to(self.device)

    def initialize_infer_attrs(self) -> Tuple[DataLoader, nn.Module]:
        d = self.build_infer_dataloader(self.conf)
        m = self.build_model(self.conf)
        return d, m

    def _build_patchers(self, knockoff=False) -> Dict[str, Patcher]:

        model = maybe_strip_ddp(self.vanilla_sae_model)
        model = copy.deepcopy(model)
        if knockoff is True:
            for name, sae in model.eval().sae_per_name.items():
                name = GroupedSAE.transform_name(name, reverse=True)
                rank_indices = self.get_column_indices_from_percentile_range(name)
                sae.set_knockoff_columns(column_indices=rank_indices)

        built_patchers = {
            (t_name := model.transform_name(name, reverse=True)): SAEPatcher(
                t_name, sae
            )
            for name, sae in model.eval().sae_per_name.items()
        }

        return built_patchers

    def infer_with_sae_knockoff_patchers(self, knockoff=False) -> Dict[str, Any]:
        records_dict = {}
        built_patchers = self._build_patchers(knockoff=knockoff)
        ds = self.dataloader
        self.conf.layer_knock_off.knockoff_feature_model = False
        use_all_layers = self.conf.layer_knock_off.use_all_layers

        def run_all_patchers():

            active_knockoff_range = self.conf.layer_knock_off.active_knockoff_range
            knockoff_range_fmt = "_".join(str(x) for x in active_knockoff_range)
            records_patcher = {}
            if knockoff is True:
                prefix = f"Knockoff_{knockoff_range_fmt}/SAE_Patcher/ALL_Patchers"
            else:
                prefix = f"Vanilla_SAE_Patcher/ALL_Patchers"
            self.feature_model.conf = self.conf
            dm = FeatureInterceptor(
                self.conf, self.feature_model, patchers_per_name=built_patchers
            )
            ev = self.build_evaluator(self.conf, runner_mode="infer")
            logger.info(
                f"Evaluation for detection in VLM with knockoff range: {active_knockoff_range} and all patcher ({built_patchers.keys()})."
            )
            _records_patcher = inference_on_dataset(
                dm, ds, ev, fwd_kwargs={"patch": True}
            )
            for metric in _records_patcher:
                for submetric in _records_patcher[metric]:
                    records_dict[f"{prefix}/{metric}_{submetric}"] = _records_patcher[
                        metric
                    ][submetric]

        def run_individual_patchers():
            active_knockoff_range = self.conf.layer_knock_off.active_knockoff_range
            knockoff_range_fmt = "_".join(str(x) for x in active_knockoff_range)
            if knockoff is True:
                prefix = f"Knockoff_{knockoff_range_fmt}/SAE_Patcher"
            else:
                prefix = f"Vanilla_SAE_Patcher"
            self.feature_model.conf = self.conf
            for name, _single_patcher in built_patchers.items():
                records_patcher = {}
                if self.conf.layer_knock_off.knockoff_layer_selection.get(name, False):
                    dm = FeatureInterceptor(
                        self.conf,
                        self.feature_model,
                        patchers_per_name={name: _single_patcher},
                    )
                    ev = self.build_evaluator(self.conf, runner_mode="infer")
                    logger.info(
                        f"Evaluation for detection in VLM with knockoff range: {active_knockoff_range} and patcher ({name})."
                    )
                    _records_patcher = inference_on_dataset(
                        dm, ds, ev, fwd_kwargs={"patch": True}
                    )
                    for metric in _records_patcher:
                        for submetric in _records_patcher[metric]:
                            records_dict[f"{prefix}/{name}/{metric}_{submetric}"] = (
                                _records_patcher[metric][submetric]
                            )
            # return records_dict

        if use_all_layers is True:
            run_all_patchers()
        elif use_all_layers is False:
            run_individual_patchers()
        elif use_all_layers == "both":
            run_all_patchers()
            run_individual_patchers()
        else:
            raise ValueError(f"Invalid value for use_all_layers: {use_all_layers}")

        return records_dict

    def infer_with_knockoff_in_feat_model(self, knockoff=True) -> Dict[str, Any]:

        # update blueglass config with the current knockoff relevant information
        records_dict = {}
        model = maybe_strip_ddp(self.vanilla_sae_model)
        sae_pactchers = model.eval().sae_per_name.keys()
        built_patchers = self._build_patchers()
        ds = self.dataloader
        use_all_layers = self.conf.layer_knock_off.use_all_layers
        self.conf.layer_knock_off.knockoff_feature_model = True

        def run_column_reduction_all_layers():
            active_knockoff_range = self.conf.layer_knock_off.active_knockoff_range
            knockoff_range_fmt = "_".join(str(x) for x in active_knockoff_range)
            self.conf.layer_knock_off.active_knockoff_layer_name = "all"
            # self.conf.layer_knock_off.active_use_all_layers_mode = True
            self.feature_model.conf = self.conf
            dm = copy.deepcopy(self.feature_model)
            ev = self.build_evaluator(self.conf, runner_mode="infer")
            logger.info(
                f"Evaluation for detection in VLM with knockoff range in FeatModel: {active_knockoff_range} using all sae's column ranks: {built_patchers.keys()}."
            )
            _records_patcher = inference_on_dataset(dm, ds, ev)
            for metric in _records_patcher:
                for submetric in _records_patcher[metric]:
                    records_dict[
                        f"KnockfOff_{knockoff_range_fmt}/FeatModel/All_Layers/{metric}_{submetric}"
                    ] = _records_patcher[metric][submetric]

        def run_column_reduction_per_layer():
            active_knockoff_range = self.conf.layer_knock_off.active_knockoff_range
            knockoff_range_fmt = "_".join(str(x) for x in active_knockoff_range)
            for name, _ in built_patchers.items():
                if self.conf.layer_knock_off.knockoff_layer_selection.get(name, False):
                    self.conf.layer_knock_off.active_knockoff_layer_name = name
                    # self.conf.layer_knock_off.active_use_all_layers_mode = True
                    self.feature_model.conf = self.conf

                    dm = copy.deepcopy(self.feature_model)
                    ev = self.build_evaluator(self.conf, runner_mode="infer")
                    logger.info(
                        f"Evaluation for detection in VLM with knockoff range in FeatModel: {active_knockoff_range} using sae's column ranks: {name}."
                    )
                    _records_patcher = inference_on_dataset(dm, ds, ev)
                    for metric in _records_patcher:
                        for submetric in _records_patcher[metric]:
                            records_dict[
                                f"KnockfOff_{knockoff_range_fmt}/FeatModel/{name}/{metric}_{submetric}"
                            ] = _records_patcher[metric][submetric]

        if use_all_layers is True:
            run_column_reduction_all_layers()

        elif use_all_layers is False:
            run_column_reduction_per_layer()

        elif use_all_layers == "both":
            run_column_reduction_all_layers()
            run_column_reduction_per_layer()

        else:
            raise ValueError(f"Invalid value for use_all_layers: {use_all_layers}")
        return records_dict

    def resolve_enabled_sae_patchers(self) -> None:
        knockoff_layer_selection = self.conf.layer_knock_off.knockoff_layer_selection
        model = self.vanilla_sae_model
        model = maybe_strip_ddp(model)
        sae_per_name = [
            model.transform_name(name, reverse=True)
            for name in model.eval().sae_per_name.keys()
        ]
        new_knockoff_layer_selection = {
            name: True
            for name in sae_per_name
            if int(name.split(".")[0].replace("layer_", "")) in knockoff_layer_selection
        }
        active_patterns = {pattern.value for pattern in self.conf.feature.patterns}
        new_knockoff_layer_selection = {
            name: any(p in name for p in active_patterns) for name in sae_per_name
        }
        self.conf.layer_knock_off.knockoff_layer_selection = (
            new_knockoff_layer_selection
        )

    def infer_run_step(self) -> Dict[str, Any]:
        """
        Perform a single inference pass over the entire dataset using three models:
        (1) the vanilla model (run once),
        (2) the model with SAEs patched into the specified layer, and
        (3) the model with ranked components knocked off in the transformer block.
        Return the output records for all three runs.
        """

        records_patcher = {}
        if self.vanilla_metrics is None:
            _records_patcher = {}
            self.conf.layer_knock_off.knockoff_feature_model = False
            self.feature_model.conf = self.conf
            """
            running vanilla evaluation only once for the entire run
            """
            ds = self.dataloader
            ev = self.build_evaluator(self.conf, runner_mode="infer")
            logger.info("Evaluation for detection in VLM (vanilla).")
            vanilla_records_patcher = inference_on_dataset(self.feature_model, ds, ev)

            # self.vanilla_metrics = {**vanilla_records_patcher, **test_patcher}
            knockoff_range_fmt = "_".join(
                str(x) for x in self.conf.layer_knock_off.active_knockoff_range
            )
            for metric in vanilla_records_patcher.keys():
                for _metric_ in vanilla_records_patcher[metric].keys():
                    _records_patcher[
                        f"vanilla/{knockoff_range_fmt}_{metric}_{_metric_}"
                    ] = vanilla_records_patcher[metric][_metric_]
            vanilla_metrics = {**_records_patcher}
            self.vanilla_metrics = vanilla_metrics
        else:
            records_patcher.update(self.vanilla_metrics)
        logger.info(
            "Evaluation for detection in VLM using sae patchers and knockoff ranges."
        )
        infer_patcher = self.infer_with_sae_knockoff_patchers(knockoff=True)

        logger.info(
            "Evaluation for detection in VLM using knockoff ranges directly in the model."
        )
        infer_knockoff = self.infer_with_knockoff_in_feat_model()

        records_patcher["infer_metrics"] = {
            **self.vanilla_metrics,
            **infer_patcher,
            **infer_knockoff,
        }
        return records_patcher

    def infer_knockoff(self) -> None:
        if self.step % self.eval_knockoff_period != 0:
            return None
        self.dataloader, self.vanilla_sae_model = self.initialize_infer_attrs()
        self.resolve_enabled_sae_patchers()
        records_column_ranks = self.sae_decoder_column_rank()

        knockoff_range = self.conf.layer_knock_off.knockoff_range
        for self.infer_step, _knockoff_range in enumerate(knockoff_range):
            self.conf.layer_knock_off.active_knockoff_range = _knockoff_range

            records_dict = self.infer_run_step()

            if self.infer_step == 0:
                records_dict["infer_metrics"].update(records_column_ranks)
            self.register_infer_metrics(records_dict, metric_mode="infer")

            del records_dict
            torch.cuda.empty_cache()
            gc.collect()
            if self.infer_step % self.logs_period == 0:
                logger.info(
                    f"Processed at {self.step}: {self.infer_step+1} / {len(self.conf.layer_knock_off.knockoff_range)}"
                )
        self.vanilla_metrics = None

    def register_infer_metrics(
        self, records_dict: Dict[str, Any], metric_mode: str = "infer"
    ) -> None:

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
            self.process_records(gathered_records_dict, metric_mode=metric_mode)
        )

        if self.conf.experiment.use_wandb:
            ranges = self.conf.layer_knock_off.knockoff_range
            df = pd.DataFrame(ranges, columns=["start", "end"])
            table = wandb.Table(dataframe=df)
            wandb.log(
                {
                    **losses_dict,
                    **metric_dict,
                    **extras_dict,
                    **visual_metric_dict,
                    "knockoff_range": wandb.plot.scatter(
                        table, x="start", y="end", title="Knockoff Percentile Ranges"
                    ),
                },
                step=self.step,
            )

        logger.info(
            f"iter: {self.infer_step:0>6}/{len(self.conf.layer_knock_off.knockoff_range)}. "
            f"metric_fitness: {metric_dict['metric_fitness']:.4f}. "
        )

    def sae_decoder_column_rank(self) -> None:
        model = copy.deepcopy(self.vanilla_sae_model)
        model = maybe_strip_ddp(model)
        records = {}
        column_rank_indices = {}
        save_file = os.path.join(
            self.conf.experiment.output_dir, "sae_decoder_column_rank"
        )
        os.makedirs(save_file, exist_ok=True)

        for name, sae in model.eval().sae_per_name.items():
            name = GroupedSAE.transform_name(name, reverse=True)
            fig, ranks = self.build_sae_column_ranks(
                sae,
                save_path=f"{save_file}/{name}_map.pdf",
                redn_method=self.conf.layer_knock_off.redn_method,
                random_state=42,
            )
            column_rank_indices[name] = ranks.tolist()
            records[f"visual_metrics_infer/column_ranks/{name}"] = fig
            plt.close(fig)
        self.column_rank_indices = column_rank_indices
        self.conf.layer_knock_off.column_ranks = column_rank_indices

        return records

    def build_sae_column_ranks(
        self,
        sae: nn.Module,
        save_path: str = None,
        redn_method: str = "tsne",
        random_state: int = 42,
    ) -> plt.Figure:
        """
        Projects decoder weights using UMAP and clusters them for visualization.

        Args:
            sae: The SAE module (single one).
            save_path: Optional file path to save the figure.
            clustering: 'kmeans' or 'dbscan'.
            n_clusters: Number of clusters for KMeans.
            random_state: Random seed for UMAP and KMeans.

        Returns:
            A matplotlib figure suitable for logging with wandb.Image().
        """
        decoder_weights = sae.sparse_codes.cpu().numpy().T  # [N, D]

        if redn_method.lower() == "umap":
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
            proj = reducer.fit_transform(decoder_weights)
            del reducer
        if redn_method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=random_state)
            proj = reducer.fit_transform(decoder_weights)
            del reducer

        center = np.mean(proj, axis=0)

        # Step 2: Compute distances
        distances = np.linalg.norm(proj - center, axis=1)

        # Step 3: Rank (0 = closest to center)
        ranks = distances.argsort().argsort()

        # Step 4: Bubble size (larger = farther)
        bubble_size = (ranks + 1) * 2  # +1 to avoid zero

        # Step 5: Color by rank (closer = darker, farther = brighter)
        colors = ranks

        # Step 6: Plot
        # Use OO-style plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            s=bubble_size,
            c=colors,
            cmap="viridis",
            edgecolors="k",
            alpha=0.8,
            linewidth=0.5,
        )
        ax.scatter(center[0], center[1], c="red", s=120, marker="x", label="Center")
        ax.set_title("2D Projection (Bubble Size and Color Reflect Distance Rank)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Distance Rank (0 = closest)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        # if save_path:
        #     fig.savefig(save_path, dpi=300)

        return fig, ranks

    def get_column_indices_from_percentile_range(self, name) -> List[int]:
        # Ensure range is ordered [low, high]
        start_percent, end_percent = sorted(
            self.conf.layer_knock_off.active_knockoff_range
        )
        columns_ranks = self.conf.layer_knock_off.column_ranks[name]

        # Convert percentages to absolute indices
        total = len(columns_ranks)
        start_idx = int((start_percent / 100) * total)
        end_idx = int((end_percent / 100) * total)

        # Slice the band
        return columns_ranks[start_idx:end_idx]

    def build_optimizer(
        self,
        conf: BLUEGLASSConf,
        model: nn.Module,
    ) -> Optimizer:
        raise NotImplementedError(
            "Optimizer is not supported for the decoder cluster runner."
        )

    def build_test_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        raise NotImplementedError(
            "Test dataloader is not supported for the decoder cluster runner."
        )

    def build_train_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        raise NotImplementedError(
            "Train dataloader is not supported for the decoder cluster runner."
        )

    def train(self) -> None:
        raise NotImplementedError(
            "Train method is not supported for the layer knockoff runner."
        )

    def test(self) -> None:
        raise NotImplementedError(
            "Test method is not supported for the layer knockoff runner."
        )

    def register_metrics(
        self, records_dict: Dict[str, Any], metric_mode: str = "test"
    ) -> None:
        if self.step % self.logs_period != 0:
            return None

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
            self.process_records(gathered_records_dict, metric_mode=metric_mode)
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
            self.checkpoint(force_save=True)
        else:
            metric_dict["metric_fitness"] = self.best_tracker.best()

        lrs_dict = {
            f"lr/pg_{i}": group["lr"]
            for i, group in enumerate(self.optimizer.param_groups)
        }
        if self.conf.experiment.use_wandb:
            if len(visual_metric_dict) > 0:
                visual_metric_dict = {
                    k: wandb.Image(v) for k, v in visual_metric_dict.items()
                }
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
