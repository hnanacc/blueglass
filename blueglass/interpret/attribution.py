# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
import math
import numpy as np
import polars as po
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm.contrib.concurrent import process_map

import torch
from blueglass.configs import BLUEGLASSConf
from blueglass.utils.logger_utils import setup_blueglass_logger
from blueglass.third_party.detectron2.data import MetadataCatalog
from .base import Interpreter

from typing import Dict, Any, Optional

from IPython import embed

logger = setup_blueglass_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetAttribution(Interpreter):
    def __init__(self, conf: BLUEGLASSConf, feature_dim: int, save_path: str):
        super().__init__(conf)
        self.feature_dim = feature_dim
        self.latents_dim = conf.sae.expansion_factor * self.feature_dim

        self.required_frame_attrs = {
            "image_id",
            "infer_id",
            "heads_id",
            "token_id",
            "pred_cls",
            "pred_box",
            "filename",
            "latents_id",
            "activation",
        }

        self.dataset_pth = MetadataCatalog.get(conf.dataset.infer).gt_images_path
        self.class_names = MetadataCatalog.get(conf.dataset.infer).thing_classes
        self.num_classes = len(self.class_names)

        self.nr_samples_per_latent = conf.interp.sae_n_samples_per_latent
        self.tokens_seen_until_now = 0
        self.top_tokens_per_latent: Optional[po.DataFrame] = None
        self.sum_latents_per_class: Optional[po.DataFrame] = None
        self.cnt_latents_per_class = np.zeros((self.num_classes,))

        self.save_path = save_path
        self.plots_dir = osp.join(self.save_path, "figures")
        self.attrb_dir = osp.join(self.save_path, "attribution")
        self.num_samples_per_row = 4

        if not osp.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        if not osp.exists(self.attrb_dir):
            os.makedirs(self.attrb_dir)

    def preprocess(
        self, batched_inputs: Dict[str, Any], batched_outputs: Dict[str, Any]
    ) -> po.DataFrame:
        """Convert IO into dataframe for easier processing."""

        assert "token_id" in batched_inputs, "Expected token ids in inputs."
        batch_size = len(batched_inputs["token_id"])

        self.tokens_seen_until_now += batch_size

        expanded_inps = {
            name: np.asarray(item).repeat(self.latents_dim, axis=0)
            for name, item in batched_inputs.items()
            if name in self.required_frame_attrs
        }

        activations = batched_outputs["prep_interims"]
        assert (
            activations.ndim == 2
            and activations.shape[0] == batch_size
            and activations.shape[1] == self.latents_dim
        ), "Unexpected shape for activations."

        expanded_outs = {
            "latents_id": np.arange(0, self.latents_dim)[np.newaxis, ...]
            .repeat(batch_size, axis=0)
            .flatten(),
            "activation": batched_outputs["prep_interims"]
            .detach()
            .cpu()
            .numpy()
            .flatten(),
        }

        expanded_merg = {**expanded_inps, **expanded_outs}

        assert (
            len({len(item) for item in expanded_merg.values()}) == 1
        ), "Expected all attributes  of merged dict to be of same size."

        return po.DataFrame(expanded_merg)

    def process_top_tokens_per_latent(self, frame: po.DataFrame):
        # reorder columns for easier concatenation later.
        frame = frame.clone().select(["latents_id", po.all().exclude("latents_id")])

        # merge current frame with top values until now.
        proc_frame = (
            self.top_tokens_per_latent.extend(frame)
            if self.top_tokens_per_latent is not None
            else frame
        )

        # update the top tokens per latents.
        self.top_tokens_per_latent = (
            proc_frame.lazy()
            .group_by("latents_id")
            .agg(po.all().top_k_by("activation", self.nr_samples_per_latent))
            .explode(po.all().exclude("latents_id"))
            .collect()
        )

    def process_top_latents_per_class(self, frame: po.DataFrame):
        # reorder and limit columns for concatenation later.
        frame = frame.clone().select(["latents_id", "pred_cls", "activation"])

        # keep count of tokens per class, as the tokens are
        # expanded by num_latents we divide to obtain the count.
        cnts_frame = frame.group_by("pred_cls").len(name="pred_cnt")
        index = cnts_frame.select("pred_cls").to_numpy().squeeze()
        value = cnts_frame.select("pred_cnt").to_numpy().squeeze() // self.latents_dim
        self.cnt_latents_per_class[index] += value

        # merge current frame with gather values until now.
        proc_frame = (
            self.sum_latents_per_class.extend(frame)
            if self.sum_latents_per_class is not None
            else frame
        )

        # update sum of activations per class and latent.
        self.sum_latents_per_class = proc_frame.group_by(
            ["latents_id", "pred_cls"]
        ).sum()

    def process(self, batched_inputs: Dict[str, Any], batched_outputs: Dict[str, Any]):
        frame = self.preprocess(batched_inputs, batched_outputs)
        self.process_top_tokens_per_latent(frame)
        self.process_top_latents_per_class(frame)

    def make_latents_distribution_graph(self):
        logger.info("Processing latents distribution graph.")
        assert (
            self.sum_latents_per_class is not None
        ), "No processed records for top_latents_per_class."

        # create a grid of all latents cross pred_cls as
        # some values may be missing in the predictions.
        latents_seq = po.int_range(0, self.latents_dim, eager=True).alias("latents_id")
        classes_seq = po.int_range(0, self.num_classes, eager=True).alias("pred_cls")
        grid_frame = po.LazyFrame(latents_seq).join(
            po.LazyFrame(classes_seq), how="cross"
        )

        # combine frames and fill activations of missing
        # combinations with zero values. note, it's lazy.
        proc_frame = (
            grid_frame.join(
                self.sum_latents_per_class.lazy(),
                on=["latents_id", "pred_cls"],
                how="left",
            )
            .with_columns(po.col("activation").fill_null(0))
            .collect()
        )

        # convert to pivoted structure for heatmap.
        pivoted = proc_frame.pivot(
            on="latents_id", index="pred_cls", values="activation"
        )
        heatmap = pivoted.select(po.exclude("pred_cls")).to_numpy()

        assert (
            heatmap.shape[0] == self.cnt_latents_per_class.shape[0]
        ), "shape mismatch in heatmap and counts."

        # convert sum activations to mean activations.
        heatmap = heatmap / self.cnt_latents_per_class[..., np.newaxis]

        # prepare labels for heatmap.
        latents = latents_seq.to_numpy()
        classes = classes_seq.to_numpy()

        fsw, fsh = max(20, len(latents) / 200), max(10, len(classes) / 5)

        fig, ax = plt.subplots(figsize=(fsw, fsh), dpi=300)
        im = ax.imshow(
            heatmap,
            aspect="auto",
            interpolation="none",
            cmap="cividis",
            origin="lower",
            vmin=heatmap.min(),
            vmax=heatmap.max(),
        )

        xticks_steps = np.arange(0, len(latents), 100)
        ax.set_xticks(latents[xticks_steps])
        ax.set_xticklabels(latents[xticks_steps], rotation=90, ha="right")
        ax.set_yticks(classes)
        ax.set_yticklabels(self.class_names)

        ax.set_xlabel("SAE Latent ID")
        ax.set_ylabel("Class ID")
        ax.set_title("Mean Activation Heatmap: Classes vs Latent Features")

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Mean Activation", rotation=270, labelpad=15)

        plt.tight_layout(pad=1)
        path = osp.join(self.plots_dir, "latents_vs_class_distribution.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved latent distribution graph at {path}.")

    def prepare_normalized_path(self, path: str):
        # TODO: fix this later in the BlueLens datasets.
        return osp.join(self.dataset_pth, f"{path.split('/')[-1]}")

    def plot_im_with_bbox(self, row: Dict[str, Any]):
        im = Image.open(self.prepare_normalized_path(row["filename"])).convert("RGB")
        ImageDraw.Draw(im).rectangle(row["pred_box"], outline="red", width=2)
        im.thumbnail((224, 224), Image.Resampling.LANCZOS)
        return im

    def make_top_tokens_grid(self, args):
        (latents_id,), frame = args
        assert isinstance(latents_id, int), "Expected latents_id to be int."
        assert isinstance(frame, po.DataFrame), "Expected frame to be a pd.DataFrame."

        assert (
            len(frame) <= self.nr_samples_per_latent
        ), "Frame has more samples than expected."

        n_rows = self.num_samples_per_row
        n_cols = math.ceil(len(frame) / self.num_samples_per_row)

        assert n_rows * n_cols >= len(frame), "Frame has more rows than grid space."

        fs_w, fs_h = n_rows * 5, n_cols * 5
        fig = plt.subplots(n_rows, n_cols, figsize=(fs_w, fs_h), dpi=300)[0]
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for ax, row in zip(fig.axes, frame.iter_rows(named=True)):
            ax.axis("off")
            ax.imshow(self.plot_im_with_bbox(row))
            ax.set_title(
                f"img: {row['image_id']}"
                f"tok: {row['token_id']}"
                f"cls: {self.class_names[row['pred_cls']].lower()}",
                fontsize=8,
            )

        plt.tight_layout(pad=0.5)
        fig.savefig(
            osp.join(self.attrb_dir, f"latent_{latents_id:0>8}.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    def interpret(self):
        assert (
            self.sum_latents_per_class is not None
        ), "No processed records for top_latents_per_class."
        assert (
            self.top_tokens_per_latent is not None
        ), "No processed records for top_latents_per_class."

        # save processed data for later use.
        self.top_tokens_per_latent.write_parquet(
            osp.join(self.save_path, "top_tokens_per_latent.parquet")
        )
        self.sum_latents_per_class.write_parquet(
            osp.join(self.save_path, "sum_latents_per_class.parquet")
        )

        # visualize latents distribution.
        self.make_latents_distribution_graph()

        # iterate through groups of latents and plot the tokens.
        process_map(
            self.make_top_tokens_grid,
            self.top_tokens_per_latent.group_by("latents_id"),
            desc="Prepare Token Grid Per Latent",
            total=self.latents_dim,
        )
