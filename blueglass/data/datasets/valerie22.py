# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog
from blueglass.third_party.detectron2.data.datasets.coco import load_coco_json

DATASET_DIR = os.environ.get("DATASET_DIR")

VALERIE22_IMAGE_ROOT = f"{DATASET_DIR}/VALERIE22/data"
VALERIE22_LABEL_ROOT = f"{DATASET_DIR}/VALERIE22/data/cocofied_annotations" + "/{}"

SPLITS = [
    ("valerie22_train", "instances_train.json"),
    ("valerie22_val", "instances_val.json"),
    ("valerie22_mini", "instances_mini.json"),
]


def custom_load_coco_json(labels_path: str, images_path: str, name: str):
    dicts = load_coco_json(labels_path, images_path, name)

    cid_to_cname = MetadataCatalog.get(name).thing_classes

    for d in dicts:
        d["annotations"] = [
            {"object_description": cid_to_cname[an["category_id"]], **an}
            for an in d["annotations"]
        ]

    return dicts


def register_valerie22(args):
    for name, an_fname in SPLITS:
        gt_images_path = VALERIE22_IMAGE_ROOT
        gt_labels_path = VALERIE22_LABEL_ROOT.format(an_fname)

        valerie22_meta = {
            "thing_classes": ["person"],
            "thing_colors": [(220, 20, 60)],
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
        MetadataCatalog.get(name).set(**valerie22_meta)
