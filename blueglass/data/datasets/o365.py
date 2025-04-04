# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0
import os
from blueglass.third_party.d2.data.datasets.coco import load_coco_json
from blueglass.third_party.d2.data.datasets.builtin_meta import _get_coco_instances_meta
from blueglass.third_party.d2.data import DatasetCatalog, MetadataCatalog


DATASET_DIR = os.environ.get("DATASET_DIR")

COCO_IMGS_PATH = f"{DATASET_DIR}/Objects365/images" + "/{}"
COCO_ANNO_PATH = f"{DATASET_DIR}Objects365/annotations" + "/{}"

COCO_ANNO_MINI_PATH = f"{DATASET_DIR}/Objects365/mini/annotations" + "/{}"

SPLITS = [
    ("o365_train", "train", "objects365_train.json"),
    ("o365_val", "val", "objects365_val.json"),
    ("o365_minitrain", "train", "objects365_train_mini.json"),
    ("o365_minival", "val", "objects365_val_mini.json"),
    # ("coco_minival", "val2017", "instances_minival2017.json"),
    # ("coco_mini", "val2017", "instances_mini2017.json"),
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


def register_objects365(args):
    for name, im_dname, an_fname in SPLITS:
        gt_images_path = COCO_IMGS_PATH.format(im_dname)
        if "mini" in name:
            gt_labels_path = COCO_ANNO_MINI_PATH.format(an_fname)
        else:
            gt_labels_path = COCO_ANNO_PATH.format(an_fname)

        coco_meta = {
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
