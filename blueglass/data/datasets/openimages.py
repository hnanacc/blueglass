# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from blueglass.utils.logger_utils import setup_blueglass_logger
import pandas as pd
from tqdm import tqdm
from blueglass.third_party.detectron2.data.datasets.coco import load_coco_json
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog

DATASET_DIR = os.environ.get("DATASET_DIR")

logger = setup_blueglass_logger(__name__)


OI_ROOT = f"{DATASET_DIR}/openimages"

OI_IMGS_TMPL = os.path.join(OI_ROOT, "images", "{}")
OI_ANNO_TMPL = os.path.join(OI_ROOT, "annotations", "cocofied_oi6_{}.json")
OI_CLASSFILE = os.path.join(OI_ROOT, "metadata", "oi6-classnames.csv")


SPLITS = [
    ("openimages_train", "train", "train"),
    ("openimages_val", "validation", "validation"),
    ("openimages_mini", "validation", "mini"),
    ("openimages_train_s0", "train", "shard_0"),
    ("openimages_train_s1", "train", "shard_1"),
    ("openimages_train_s2", "train", "shard_2"),
    ("openimages_train_s3", "train", "shard_3"),
    ("openimages_train_s4", "train", "shard_4"),
    ("openimages_train_s5", "train", "shard_5"),
    ("openimages_train_s6", "train", "shard_6"),
    ("openimages_train_s7", "train", "shard_7"),
    ("openimages_train_s8", "train", "shard_8"),
    ("openimages_train_s9", "train", "shard_9"),
    ("openimages_train_sa", "train", "shard_a"),
    ("openimages_train_sb", "train", "shard_b"),
    ("openimages_train_sc", "train", "shard_c"),
    ("openimages_train_sd", "train", "shard_d"),
    ("openimages_train_se", "train", "shard_e"),
    ("openimages_train_sf", "train", "shard_f"),
]


def custom_load_coco_json(labels_path: str, images_path: str, name):
    dicts = load_coco_json(labels_path, images_path, name)
    cnames = MetadataCatalog.get(name).thing_classes

    for d in tqdm(dicts, desc="add descriptions to samples"):
        d["annotations"] = [
            {"object_description": cnames[an["category_id"]], **an}
            for an in d["annotations"]
        ]

    return dicts


def openimages_classes():
    cns = pd.read_csv(OI_CLASSFILE).DisplayName
    cns = ["_".join(stem.strip() for stem in cn.lower().split()) for cn in cns]
    return cns


def register_openimages(args):
    for name, im_name, an_name in SPLITS:
        gt_images_path = OI_IMGS_TMPL.format(im_name)
        gt_labels_path = OI_ANNO_TMPL.format(an_name)

        meta = {
            "thing_classes": openimages_classes(),
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
        MetadataCatalog.get(name).set(**meta)
