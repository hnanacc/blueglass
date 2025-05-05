# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import gc
import copy
import wandb
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import Dict, Any, List, Tuple, Union
from functools import lru_cache
from fvcore.common.checkpoint import Checkpointer
from collections import defaultdict, ChainMap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import umap.umap_ as umap
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


class DecoderClusterRunner(SAERunner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.conf = conf
        self.runner_name = str(self.conf.runner.name)
        self.runner_model_name = str(self.conf.model.name)

        self.step = 1
        self.max_steps = conf.runner.max_steps
        self.warmup_steps = conf.runner.warmup_steps

        self.eval_period = conf.runner.eval_period
        self.logs_period = conf.runner.logs_period
        self.precision = getattr(torch, conf.runner.precision)

        self.device = DEVICE
        self.feature_model = self._frozen(build_model(conf))

        self.vanilla_metrics = None
        self.cluster_knockoff_indices = None

        assert isinstance(self.precision, torch.dtype), "Invalid precision."

        assert (
            self.eval_period >= self.logs_period
        ), "invalid eval period and logs period, logs must be smaller than eval."

        assert (
            self.eval_period % self.logs_period == 0
        ), "invalid eval period and logs period, must be divisible."

    def build_infer_dataloader(self, conf: BLUEGLASSConf) -> DataLoader:
        return build_test_dataloader(
            conf.dataset.test, conf.dataset.batch_size, conf.num_data_workers
        )

    def prepare_filter_scheme(
        self, conf: BlockingIOError, remove_io: bool = True
    ) -> str:
        patterns = conf.feature.patterns
        if remove_io and FeaturePattern.IO in patterns:
            patterns.remove(FeaturePattern.IO)
        patterns = "|".join(patterns) if len(patterns) > 0 else r"\w+"

        subpatns = conf.feature.sub_patterns
        subpatns = "|".join(subpatns) if len(subpatns) > 0 else r"\w+"

        layerids = conf.feature.layer_ids
        layerids = [str(li) for li in layerids]
        layerids = "|".join(layerids) if len(layerids) > 0 else r"\d+"

        return f"layer_({layerids}).({patterns}).({subpatns})"

    def build_saes_model(self, conf) -> nn.Module:
        store_meta = FeatureDataset(
            conf,
            conf.dataset.train,
            self._prepare_model_for_store(conf),
            filter_scheme=self.prepare_filter_scheme(conf),
        ).infer_feature_meta()

        assert (
            "feature_dim_per_name" in store_meta
        ), "Feature dims not found in store meta."

        return create_ddp_model(
            GroupedSAE(conf, store_meta["feature_dim_per_name"]).to(self.device),
            broadcast_buffers=False,
        )

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

    def initialize_infer_attrs(self) -> Tuple[DataLoader, nn.Module, DatasetEvaluator]:
        """
        Initialize the model for inference.
        """
        config = load_blueglass_from_wandb(
            self.conf.sae.config_path, original_config=self.conf
        )
        filters = ["decoder_mlp"]
        patterns = config.feature.patterns
        config.feature.patterns = [
            p for p in patterns if any(f in p.value for f in filters)
        ]

        m = self.build_saes_model(config)
        ckpt = torch.load(
            self.conf.sae.checkpoint_path, map_location="cpu", weights_only=False
        )
        missing_keys, unexpected_keys = m.load_state_dict(ckpt["model"], strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logger.warning(
                f"Missing keys in state_dict: {missing_keys}. "
                f"Unexpected keys in state_dict: {unexpected_keys}."
            )
        m.eval()

        return m

    def run_step(self) -> Dict[str, Any]:
        """
        running vanilla evaluation only once for the entire run
        """

        records_cluster = self.sae_decoder_cluster_knockoff()

        records_patcher = {}
        if self.vanilla_metrics is None:
            ds = build_test_dataloader(
                self.conf.dataset.test, self.conf.dataset.batch_size
            )
            ev = self.build_evaluator(self.conf)
            logger.info("Evaluation for detection in VLM (vanilla).")
            vanilla_records_patcher = inference_on_dataset(self.feature_model, ds, ev)
            test_patcher = self.patcher_test()

            self.vanilla_metrics = {**vanilla_records_patcher, **test_patcher}

            for metric in vanilla_records_patcher.keys():
                for _metric_ in vanilla_records_patcher[metric].keys():
                    records_patcher[f"vanilla/{metric}_{_metric_}"] = (
                        vanilla_records_patcher[metric][_metric_]
                    )
        else:
            vanilla_records_patcher = self.vanilla_metrics
            for metric in vanilla_records_patcher.keys():
                for _metric_ in vanilla_records_patcher[metric].keys():
                    records_patcher[f"vanilla/{metric}_{_metric_}"] = (
                        vanilla_records_patcher[metric][_metric_]
                    )

        test_patcher = self.patcher_test()
        records_patcher["metrics"] = records_patcher | test_patcher
        return records_patcher

    def infer(self) -> Dict[str, Any]:
        self.vanilla_sae_model = self.initialize_infer_attrs()

        for self.step in range(1, self.max_steps + 1):
            records_dict = self.run_step()

            self.register_metrics(records_dict)

            del records_dict
            torch.cuda.empty_cache()
            gc.collect()

    def register_metrics(self, records_dict: Dict[str, Any]):
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

    def sae_decoder_cluster_knockoff(self) -> None:
        model = copy.deepcopy(self.vanilla_sae_model)
        model = maybe_strip_ddp(model)

        records = {}
        cluster_knockoff_indices = {}
        clustering = self.conf.runner.cluster_method
        save_file = os.path.join(
            self.conf.experiment.output_dir, "sae_decoder_cluster", clustering
        )
        os.makedirs(save_file, exist_ok=True)
        if self.step == 1:
            for name, sae in model.eval().sae_per_name.items():
                fig, _cluster_indices = self.analyze_decoder_clusters(
                    sae,
                    save_path=f"{save_file}/{name}_map.pdf",
                    clustering=clustering,
                    n_clusters=self.conf.runner.n_clusters,
                    random_state=42,
                )
                records[name] = fig
                cluster_knockoff_indices[name] = _cluster_indices
                plt.close(fig)
            self.cluster_knockoff_indices = cluster_knockoff_indices
        else:
            """
            This section handles the knock off indices of the decoder from above computed clusters in each SAE model depending on the step
            """

            for name, sae in model.eval().sae_per_name.items():
                # _cluster_indices = self.get_knockoff_indices(sae)
                cluster_knockoff_indices[name] = _cluster_indices

        self.model = model
        return records

    def analyze_decoder_clusters(
        self,
        sae: nn.Module,
        save_path: str = None,
        clustering: str = "kmeans",
        n_clusters: int = 10,
        random_state: int = 42,
        dbscan_eps: float = 0.5,
        hdbscan_min_cluster_size: int = 50,
        min_samples: int = 5,
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
        decoder_weights = sae.decoder.detach().cpu().numpy()  # [N, D]
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
        umap_proj = reducer.fit_transform(decoder_weights)
        del reducer
        # Clustering
        if clustering == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            labels = clusterer.fit_predict(umap_proj)
        elif clustering == "dbscan":
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
            labels = clusterer.fit_predict(umap_proj)
        elif clustering == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
            labels = clusterer.fit_predict(umap_proj)
        else:
            raise ValueError(f"Unknown clustering: {clustering}")

        # Map cluster_id -> row indices
        cluster_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(idx)

        # Optional: sort indices per cluster
        for k in cluster_indices:
            cluster_indices[k].sort()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            umap_proj[:, 0], umap_proj[:, 1], c=labels, cmap="tab10", s=3
        )
        ax.set_title(f"SAE Decoder Cluster Map ({clustering})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)

        return fig, cluster_indices

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
            "Train method is not supported for the decoder cluster runner."
        )

    def test(self) -> None:
        raise NotImplementedError(
            "Test method is not supported for the decoder cluster runner."
        )

    def checkpoint(self):
        raise NotImplementedError(
            "Checkpoint is not supported for the decoder cluster runner."
        )
