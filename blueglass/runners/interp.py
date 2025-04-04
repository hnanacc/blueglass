# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import shutil
from blueglass.utils.logger_utils import setup_blueglass_logger
import pickle
from collections import defaultdict
from typing import List, Dict, Any
import torch
from blueglass.runners import Runner
from blueglass.third_party.detectron2.utils.logger import log_every_n
from blueglass.third_party.detectron2.data import MetadataCatalog
from blueglass.third_party.detectron2.structures import Boxes
from blueglass.configs import BLUEGLASSConf

from blueglass.evaluation import compute_confusion_mask
from blueglass.interp import DatasetAttribution


logger = setup_blueglass_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterpretationRunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.device = DEVICE
        self.save_path = os.path.join(conf.experiment.output_dir, "assets")

        self.dataset = conf.dataset.infer
        self.dataloader = self.build_test_dataloader(conf)
        self.model = self.build_model(conf).eval()

        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path)

        self.step, self.max_steps = 0, 0

    @torch.inference_mode()
    def run_step(self, batched_inputs: List[Dict]):

        with torch.inference_mode():
            batched_outputs = self.model(batched_inputs)

        return records, batched_outputs

    def visualize_clusters_umap(self):
        for layer, samples in self.samples_per_layer.items():
            vis_path = os.path.join(self.save_path, f"clusters_{layer}.pdf")
            visualize_clusters(
                samples["features"], samples["labels"], vis_path, method="umap"
            )

    def compute_information_plane(self):
        pass

    def compute_effective_dimensions(self):
        pass

    def compute_confusion_mask(
        self,
        batched_inputs: List[Dict[str, Any]],
        features,
        batched_outputs: List[Dict[str, Any]],
    ):
        preds = features.fetch_io(IOPatterns.PRED)
        pred_box = preds["pred_box"]
        pred_cls = preds["pred_cls"]
        pred_scr = preds["pred_scr"]

        batched_outputs = [
            {
                **bo,
                "unprocessed_boxes": Boxes(pb),
                "unprocessed_clsid": pc,
                "unprocessed_score": ps,
            }
            for bo, pb, pc, ps in zip(batched_outputs, pred_box, pred_cls, pred_scr)
        ]

        return compute_confusion_mask(self.conf, batched_inputs, batched_outputs)

    def prepare_storage(self):
        self.n_classes = len(MetadataCatalog.get(self.dataset).thing_classes)
        self.n_samples_per_class = 100
        self.samples_per_class_sofar = [0] * self.n_classes
        self.samples_per_layer = defaultdict(lambda: {"features": [], "labels": []})

    def process(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_outputs: List[Dict[str, Any]],
        sequence,
    ):
        features = self.seq_processor(sequence)

        cfmask = self.compute_confusion_mask(batched_inputs, features, batched_outputs)
        labels = features.fetch_io(IOPatterns.PRED_CLS)["pred_cls"]
        tarfts = features.fetch(self.pattern_name, FeatureSubPatterns("pos_img"))

        for batch_ind, (label, cfms) in enumerate(zip(labels, cfmask)):
            if cfms.sum() == 0:
                continue
            else:
                cfms = cfms.to(device=self.device, dtype=torch.long)

            for cls_ind in range(self.n_classes):
                if self.samples_per_class_sofar[cls_ind] < self.n_samples_per_class:
                    mask = (label == cls_ind) & (cfms)

                    if mask.sum() == 0:
                        continue

                    chosen_ind = mask.nonzero()[0][0].item()

                    for layer, tarft in tarfts.items():
                        self.samples_per_layer[layer]["features"].append(
                            tarft["pos_img"][batch_ind][chosen_ind]
                        )
                        self.samples_per_layer[layer]["labels"].append(cls_ind)

                    self.samples_per_class_sofar[cls_ind] += 1

    def compute(self):
        self.visualize_clusters_umap()
        self.compute_information_plane()
        self.compute_effective_dimensions()

    def infer(self):
        max_steps = len(self.dataloader)
        self.prepare_storage()
        for self.step, batched_inputs in enumerate(self.dataloader, start=1):
            batched_sequences, batched_outputs = self.run_step(batched_inputs)
            self.process(batched_inputs, batched_outputs, batched_sequences)
            log_every_n(
                logging.INFO,
                f"Processed {self.step}/{max_steps}. "
                f"Sampled {sum(self.samples_per_class_sofar)}. "
                f"Min sampled {min(self.samples_per_class_sofar)}. "
                f"Max sampled {max(self.samples_per_class_sofar)}.",
                n=5,
            )

            if sum(self.samples_per_class_sofar) == (
                self.n_classes * self.n_samples_per_class
            ):
                logger.info("reached required samples.")
                break

        path = os.path.join(self.save_path, "sampled_features.pkl")
        with open(path, "wb") as fp:
            pickle.dump(self.samples_per_layer, fp)
        logger.info(f"saved features at path: {path}")

    def train(self):
        raise NotImplementedError("unsupported. use infer.")

    def test(self):
        raise NotImplementedError("unsupported. use infer.")
