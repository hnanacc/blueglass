# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from collections import OrderedDict
from blueglass.utils.logger_utils import setup_blueglass_logger
from itertools import chain
# from scalabel.eval.detect import evaluate_det
# from scalabel.label.io import load
# from scalabel.label.typing import Config, Frame

from blueglass.third_party.scalabel.scalabel.eval.detect import evaluate_det
from blueglass.third_party.scalabel.scalabel.label.io import load
from blueglass.third_party.scalabel.scalabel.label.typing import Config, Frame


from blueglass.third_party.detectron2.data import MetadataCatalog
from blueglass.third_party.detectron2.utils import comm
from blueglass.third_party.detectron2.utils.logger import create_small_table
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from typing import Dict, Any, List, Optional
from blueglass.structures.types import is_comm_list_of_dict

logger = setup_blueglass_logger(__name__)

CONFIG = {
    "imageSize": {"height": 720, "width": 1280},
    "attributes": [{"name": "crowd", "type": "switch", "tag": "c"}],
    "categories": [
        {"name": "human", "subcategories": [{"name": "person"}, {"name": "rider"}]},
        {
            "name": "vehicle",
            "subcategories": [
                {"name": "car"},
                {"name": "truck"},
                {"name": "bus"},
                {"name": "train"},
            ],
        },
        {
            "name": "bike",
            "subcategories": [{"name": "bicycle"}, {"name": "motorcycle"}],
        },
        {"name": "traffic light"},
        {"name": "traffic sign"},
    ],
}

class BDD100kEvaluator(DatasetEvaluator):
    def __init__(
        self, dataset_name: str, output_dir: Optional[str] = None, nprocs: int = 1
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.meta = MetadataCatalog.get(dataset_name)
        self.nprocs = nprocs

        self.gtlb_path = self.meta.gt_labels_path
        self.gtim_path = self.meta.gt_images_path

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

            self.dt_frames.append(
                {
                    "name": os.path.basename(inp["file_name"]),
                    "labels": [
                        {
                            "id": aid,
                            "category": self.classnames[clsid],
                            "score": score,
                            "box2d": {
                                "x1": bbox[0],
                                "y1": bbox[1],
                                "x2": bbox[2],
                                "y2": bbox[3],
                            },
                        }
                        for aid, (bbox, clsid, score) in enumerate(
                            zip(bboxes, clsids, scores)
                        )
                    ],
                }
            )

    def evaluate(self):
        gather_dt_frames = comm.gather(self.dt_frames, dst=0)
        if not comm.is_main_process():
            return

        assert is_comm_list_of_dict(gather_dt_frames), "unexpected comm data."

        dt_frames = [Frame(**f) for f in chain(*gather_dt_frames)]
        gt_frames = load(self.gtlb_path, nprocs=self.nprocs).frames

        r = evaluate_det(gt_frames, dt_frames, Config(**CONFIG), nproc=self.nprocs)

        overall = {k: v[1]["OVERALL"] for k, v in r.dict().items()}
        print(f"\n{create_small_table(overall)}\n")
        overall_metrics = {f"bbox/{k}": v for k, v in overall.items()}
        return overall_metrics
