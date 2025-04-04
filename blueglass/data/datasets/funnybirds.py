# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog
from blueglass.third_party.detectron2.data.datasets.coco import load_coco_json

DATASET_DIR = os.environ.get("DATASET_DIR")

FUNNY_IMGS_PATH = f"{DATASET_DIR}/funnybirds/cocofied_10k/images"
FUNNY_ANNO_PATH = f"{DATASET_DIR}/funnybirds/cocofied_10k/annotations" + "/{}"

SPLITS = [
    (
        "funnybirds_no_intervention",
        "instances_isometric_camera.json",
    ),
    ("funnybirds_no_beak", "instances_isometric_no_beak.json"),
    ("funnybirds_no_eyes", "instances_isometric_no_eyes.json"),
    ("funnybirds_no_foot", "instances_isometric_no_foot.json"),
    ("funnybirds_no_tail", "instances_isometric_no_tail.json"),
    ("funnybirds_no_wings", "instances_isometric_no_wings.json"),
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


def register_funnybirds(args):
    for name, an_fname in SPLITS:
        gt_images_path = FUNNY_IMGS_PATH
        gt_labels_path = FUNNY_ANNO_PATH.format(an_fname)

        funnybirds_meta = {
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
        MetadataCatalog.get(name).set(**funnybirds_meta)
