# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import shutil
import json
from typing import Dict, Any, List, Optional
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import List, Dict, Any
from blueglass.third_party.ecpb.eval import evaluate_detection
from blueglass.third_party.detectron2.data import MetadataCatalog
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator

logger = setup_blueglass_logger(__name__)


class ECPersonsEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name: str,
        output_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.meta = MetadataCatalog.get(dataset_name)

        self.gt_images_path = self.meta.gt_images_path
        self.gt_labels_path = self.meta.gt_labels_ecpb_path
        self.dt_labels_path = os.path.join(output_dir, "predictions")

        if os.path.exists(self.dt_labels_path):
            logger.info("Found existing predictions dir. Purging...")
            shutil.rmtree(self.dt_labels_path)

        os.makedirs(self.dt_labels_path)

        self.classnames = self.meta.thing_classes

        self.dt_frames = []

    def reset(self):
        self.dt_frames = []

    def process(self, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]):
        for inp, out in zip(inputs, outputs):
            inst = out["instances"].to("cpu")

            bboxes = inst.pred_boxes.tensor.numpy()
            clsids = inst.pred_classes.tolist()
            scores = inst.scores.tolist()

            frames = {
                "identity": "frame",
                "children": [
                    {
                        "x0": float(bbox[0]),
                        "y0": float(bbox[1]),
                        "x1": float(bbox[2]),
                        "y1": float(bbox[3]),
                        "score": score,
                        "identity": self.classnames[clsid],
                        "orient": 0.0,
                    }
                    for bbox, clsid, score in zip(bboxes, clsids, scores)
                ],
            }

            fn = os.path.basename(inp["file_name"]).replace("png", "json")
            with open(os.path.join(self.dt_labels_path, fn), "w") as fp:
                json.dump(frames, fp)

    def evaluate(self):
        evaluate_detection(
            self.output_dir, self.dt_labels_path, self.gt_labels_path, self.dataset_name
        )
