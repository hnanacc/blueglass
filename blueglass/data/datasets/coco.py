# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from blueglass.third_party.detectron2.data.datasets.coco import load_coco_json
from blueglass.third_party.detectron2.data.datasets.builtin_meta import (
    _get_coco_instances_meta,
)
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog

DATASET_DIR = os.environ.get("DATASET_DIR")

COCO_IMGS_PATH = f"{DATASET_DIR}/COCO/coco2017" + "/{}"
COCO_ANNO_PATH = f"{DATASET_DIR}/COCO/coco2017/annotations" + "/{}"

SPLITS = [
    ("coco_train", "train2017", "instances_train2017.json"),
    ("coco_val", "val2017", "instances_val2017.json"),
    ("coco_minival", "val2017", "instances_minival2017.json"),
    ("coco_mini", "val2017", "instances_mini2017.json"),
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


def register_coco(args):
    for name, im_dname, an_fname in SPLITS:
        gt_images_path = COCO_IMGS_PATH.format(im_dname)
        gt_labels_path = COCO_ANNO_PATH.format(an_fname)

        coco_meta = {
            **_get_coco_instances_meta(),
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
        MetadataCatalog.get(name).set(**coco_meta)
