# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import Dict, List, Tuple, Any
import torch
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog
from blueglass.third_party.detectron2.structures import BoxMode, Instances, Boxes
from blueglass.structures import Descriptions

DATASET_DIR = os.environ.get("DATASET_DIR")

logger = setup_blueglass_logger(__name__)

BDD100K_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

IGNORED_CLASSES = ["other vehicle", "other person", "trailer"]

BDD100K_IMG_H = 720
BDD100K_IMG_W = 1280


def make_instance(
    boxes: List[Tuple[float, float, float, float]],
    clses: List[int],
    descs: List[str],
):
    if len(boxes) == 0:
        bxtensor = torch.empty((0, 4))
    else:
        bxtensor = torch.tensor(boxes)

    return Instances(
        (BDD100K_IMG_H, BDD100K_IMG_W),
        gt_boxes=Boxes(bxtensor),
        gt_classes=torch.tensor(clses, dtype=torch.long),
        object_descriptions=Descriptions(descs),
    )


def scalabel_to_d2(sample: Dict[str, Any]):
    if "labels" not in sample:
        return make_instance([], [], [])

    boxes, clses, descs = [], [], []

    for anno in sample["labels"]:
        clsname = anno["category"].lower()

        if clsname not in BDD100K_CLASSES:
            continue

        boxes.append(
            [
                anno["box2d"]["x1"],
                anno["box2d"]["y1"],
                anno["box2d"]["x2"],
                anno["box2d"]["y2"],
            ]
        )
        clses.append(BDD100K_CLASSES.index(clsname))
        descs.append(clsname)

    return make_instance(boxes, clses, descs)


def prepare_instances(images_path: str, labels_path: str):
    with open(labels_path) as f:
        samples = sorted(json.load(f), key=lambda d: d["name"])

    return [
        {
            "file_name": os.path.join(images_path, sample["name"]),
            "image_id": sample["name"].split(".")[0],
            "height": BDD100K_IMG_H,
            "width": BDD100K_IMG_W,
            "instances": scalabel_to_d2(sample),
        }
        for sample in samples
    ]


BDD100K_IMAGE_TMPL = f"{DATASET_DIR}/bdd100k/images/100k" + "/{}"
BDD100K_LABEL_TMPL = f"{DATASET_DIR}/bdd100k/labels/det_20" + "/det_{}.json"

SPLITS = [
    ("bdd100k_train", "train"),
    ("bdd100k_val", "val"),
    ("bdd100k_mini", "mini"),
]


def register_bdd100k(args):
    for name, split in SPLITS:
        im_path = BDD100K_IMAGE_TMPL.format(split)
        lb_path = BDD100K_LABEL_TMPL.format(split)

        meta = {
            "thing_classes": BDD100K_CLASSES,
            "gt_images_path": im_path,
            "gt_labels_path": lb_path,
            "evaluator_type": "bdd100k",
        }

        DatasetCatalog.register(
            name, lambda ip=im_path, lp=lb_path: prepare_instances(ip, lp)
        )
        MetadataCatalog.get(name).set(**meta)
