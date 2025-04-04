# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from typing import Dict
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog
from blueglass.third_party.detectron2.data.datasets.coco import load_coco_json

DATASET_DIR = os.environ.get("DATASET_DIR")

ECPERSONS_IMAGE_ROOT = f"{DATASET_DIR}/eurocity"
ECPERSONS_LABEL_ROOT = f"{DATASET_DIR}/eurocity/cocofied_annotations" + "/{}"

SPLITS = [
    (
        "ecpersons_train",
        "instances_day_train.json",
        os.path.join(ECPERSONS_IMAGE_ROOT, "ECP/day/labels/train"),
    ),
    (
        "ecpersons_val",
        "instances_day_val.json",
        os.path.join(ECPERSONS_IMAGE_ROOT, "ECP/day/labels/val"),
    ),
    (
        "ecpersons_mini",
        "instances_day_mini.json",
        os.path.join(ECPERSONS_IMAGE_ROOT, "ECP/day/labels/mini"),
    ),
]


def custom_load_coco_json(labels_path: str, images_path: str, name: str):
    dicts = load_coco_json(labels_path, images_path, name)

    cid_to_cname = MetadataCatalog.get(name).thing_classes

    for d in dicts:
        annos = [
            {"object_description": cid_to_cname[an["category_id"]], **an}
            for an in d["annotations"]
        ]
        d["annotations"] = annos

    return dicts


def register_per_split(meta: Dict, labels_path: str, images_path: str, name: str):
    MetadataCatalog.get(name).set(**meta)
    DatasetCatalog.register(
        name,
        lambda: custom_load_coco_json(labels_path, images_path, name),
    )


def register_ecpersons(args):
    for name, an_fname, ecpb_path in SPLITS:
        gt_images_path = ECPERSONS_IMAGE_ROOT
        gt_labels_path = ECPERSONS_LABEL_ROOT.format(an_fname)

        ecpersons_meta = {
            "thing_classes": ["pedestrian"],
            "thing_colors": [(220, 20, 60)],
            "gt_images_path": gt_images_path,
            "gt_labels_coco_path": gt_labels_path,
            "gt_labels_ecpb_path": ecpb_path,
            "evaluator_type": "coco",
        }

        register_per_split(ecpersons_meta, gt_labels_path, gt_images_path, name)
