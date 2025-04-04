# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from blueglass.third_party.detectron2.data.datasets.lvis import (
    load_lvis_json,
    get_lvis_instances_meta,
)
from blueglass.third_party.detectron2.data import DatasetCatalog, MetadataCatalog

DATASET_DIR = os.environ.get("DATASET_DIR")

LVIS_IMAGE_ROOT = f"{DATASET_DIR}/lvis/"
LVIS_LABEL_TMPL = f"{DATASET_DIR}/lvis" + "/{}.json"

SPLITS = [
    ("lvis_train", "lvis_v1_train"),
    ("lvis_val", "lvis_v1_val"),
    ("lvis_minival", "lvis_v1_minival"),
    ("lvis_mini", "lvis_v1_mini"),
]


def custom_load_lvis_json(labels_path: str, images_path: str, name):
    dicts = load_lvis_json(labels_path, images_path, name)
    cnames = MetadataCatalog.get(name).thing_classes

    for d in dicts:
        d["annotations"] = [
            {"object_description": cnames[an["category_id"]], **an}
            for an in d["annotations"]
        ]

    return dicts


def register_lvis(args):
    for name, an_fname in SPLITS:
        an_path = LVIS_LABEL_TMPL.format(an_fname)
        v1_meta = get_lvis_instances_meta("v1")

        DatasetCatalog.register(
            name,
            lambda an=an_path, dn=name: custom_load_lvis_json(an, LVIS_IMAGE_ROOT, dn),
        )
        MetadataCatalog.get(name).set(
            json_file=an_path, image_root=LVIS_IMAGE_ROOT, **v1_meta
        )
