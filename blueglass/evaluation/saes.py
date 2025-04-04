# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Dict, Any
from torch import Tensor
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict
from blueglass.configs import BLUEGLASSConf
from blueglass.structures.types import is_comm_dict, is_comm_list_of_dict
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from blueglass.third_party.detectron2.utils import comm
from itertools import chain
import scienceplots


class SAEEvaluator(DatasetEvaluator):
    def __init__(self, conf: BLUEGLASSConf):
        self.conf = conf
        self.metric_terms = {
            "loss_reconstr",
            "norm_l0",
            "norm_l1",
            "dead_pct",
            "dense_pct",
            "latents_dead_since",
            "latents_fire_count",
        }
        self._processed = []
        self.setup_graphics()

    def setup_graphics(self):
        try:
            plt.style.use(
                ["science", "grid", "notebook"]
            )  # Remove "notebook" if it causes issues
        except:
            plt.style.use(
                ["seaborn", "grid", "notebook"]
            )  # Fallback to a built-in style
            print("SciencePlots not installed. Using fallback style.")

    def process(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        _processed = {
            attr: item
            for attr, item in outputs.items()
            if any(term in attr for term in self.metric_terms)
        }
        self._processed.append(_processed)

    def normalize(self, tensor: Tensor):
        assert tensor.ndim == 1, "Expected one-dim tensor."
        nmin, nmax = tensor.min(), tensor.max()
        return (tensor - nmin) / nmax

    def visualize_latents_distribution(
        self, latents_fire_count: Tensor, latents_dead_since: Tensor
    ) -> Dict[str, Figure]:
        """
        Visualizes latent distributions with dummy plots.

        Args:
            latents_fire_count: Tensor of firing counts
            latents_dead_since: Tensor of dead since timestamps

        Returns:
            Dictionary containing distribution and histogram figures
        """
        try:
            # Normalize inputs
            latents_fire_count = self.normalize(
                torch.randn(100)
            )  # or torch.rand(100) for [0,1] range
            latents_dead_since = self.normalize(torch.randn(100))

            # Create figures
            dist_fig = plt.figure(figsize=(12, 10))
            hist_fig = plt.figure(figsize=(12, 6))

            # Dummy distribution plot (KDE)
            ax1 = dist_fig.add_subplot(211)
            ax1.set_title("Normalized Fire Count Distribution")
            ax1.plot([0, 1], [0, 1], "r--")  # Simple diagonal line
            ax1.set_ylabel("Density")

            ax2 = dist_fig.add_subplot(212)
            ax2.set_title("Normalized Dead Since Distribution")
            ax2.plot([0, 1], [1, 0], "b--")  # Inverse diagonal line
            ax2.set_ylabel("Density")

            # Dummy histogram
            ax3 = hist_fig.add_subplot(111)
            ax3.set_title("Combined Latent Distribution")
            ax3.hist([0.2, 0.5, 0.5, 0.6, 0.8], bins=10, alpha=0.5)  # Simple histogram
            ax3.set_xlabel("Normalized Values")
            ax3.set_ylabel("Frequency")

            plt.tight_layout()

            return {"latents_dist": dist_fig, "latents_hist": hist_fig}

        except Exception as e:
            plt.close("all")
            raise RuntimeError(f"Visualization failed: {str(e)}")

    def evaluate(self) -> Dict[str, Any]:
        comm.synchronize()
        gathered = comm.gather(self._processed)
        assert is_comm_list_of_dict(gathered), "Unexpected comm type."
        gathered = list(chain.from_iterable(gathered))
        if not comm.is_main_process():
            return {}

        reduced = defaultdict(list)
        for records in gathered:
            for name, item in records.items():
                reduced[name].append(item)

        latents_fire_count = {}
        latents_dead_since = {}

        for k in list(reduced.keys()):
            if "latents_fire_count" in k:
                latents_fire_count[k] = reduced.pop(k)
            elif "latents_dead_since" in k:
                latents_dead_since[k] = reduced.pop(k)

        visuals = self.visualize_latents_distribution(
            latents_fire_count, latents_dead_since
        )
        visuals = {f"visual_metrics/{k}": v for k, v in visuals.items()}
        # TODO Fix visual plots
        visuals = {}
        # change the name to prevent conflict with train
        # losses while parsing for metrics in inference.
        for key in list(reduced.keys()):
            if "loss_reconstr" in key:
                new_key = key.replace("loss_reconstr", "reconstr_err")
                reduced[new_key] = reduced.pop(key)

        # normlize values across ranks.
        reduced = {
            name: torch.stack([t.cpu() for t in item]).mean().item()
            for name, item in reduced.items()
        }
        return {**reduced, **visuals}
