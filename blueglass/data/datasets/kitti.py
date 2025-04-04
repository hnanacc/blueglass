# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from blueglass.third_party.detectron2.data.datasets.coco import load_coco_json
from blueglass.third_party.detectron2.data.datasets.builtin_meta import (
    _get_coco_instances_meta,
)
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog

DATASET_DIR = os.environ.get("DATASET_DIR")

KITTI_ROOT = f"{DATASET_DIR}/KITTI/cocofied"
KITTI_IMGS_PATH = os.path.join(KITTI_ROOT, "images")
KITTI_ANNO_TMPL = os.path.join(KITTI_ROOT, "annotations", "{}.json")

KITTI_CATEGORIES = ["car", "pedestrian", "cyclist"]

SPLITS = [
    ("kitti_train", "instances_train"),
    ("kitti_val", "instances_val"),
    ("kitti_mini", "instances_mini"),
]


def custom_load_coco_json(labels_path: str, images_path: str, name):
    dicts = load_coco_json(labels_path, images_path, name)
    cnames = MetadataCatalog.get(name).thing_classes

    for d in dicts:
        d["annotations"] = [
            {"object_description": cnames[an["category_id"]], **an}
            for an in d["annotations"]
        ]

    return dicts


def register_kitti(args):
    for name, annos_fname in SPLITS:
        gt_images_path = KITTI_IMGS_PATH
        gt_labels_path = KITTI_ANNO_TMPL.format(annos_fname)

        kitti_meta = {
            "thing_classes": KITTI_CATEGORIES,
            "gt_images_path": gt_images_path,
            "gt_labels_path": gt_labels_path,
            "evaluator_type": "coco",
        }

        DatasetCatalog.register(
            name,
            lambda lp=gt_labels_path, ip=gt_images_path, nm=name: custom_load_coco_json(
                lp, ip, nm
            ),
        )
        MetadataCatalog.get(name).set(**kitti_meta)
