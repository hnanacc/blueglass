# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
from typing import List, Tuple
from .defaults import Dataset, Evaluator

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(os.getcwd(), "weights"))
FEATURE_DIR = os.environ.get("FEATURE_DIR", osp.join(os.getcwd(), "bluelens"))
MODELSTORE_CONFIGS_DIR = osp.join(os.getcwd(), "blueglass", "modeling", "modelstore")
MODELSTORE_MMDET_CONFIGS_DIR = osp.join(
    os.getcwd(), "blueglass", "third_party", "mmdet", "configs"
)


DATASETS_AND_EVALS: List[Tuple[str, Dataset, Dataset, Evaluator]] = [
    (
        "funnybirds",
        Dataset.FUNNYBIRDS_NO_INTERVENTION,
        Dataset.FUNNYBIRDS_NO_BEAK,
        Evaluator.COCO,
    ),
    ("valerie", Dataset.VALERIE22_TRAIN, Dataset.VALERIE22_VAL, Evaluator.COCO),
    ("ecpersons", Dataset.ECPERSONS_TRAIN, Dataset.ECPERSONS_VAL, Evaluator.COCO),
    ("kitti", Dataset.KITTI_TRAIN, Dataset.KITTI_VAL, Evaluator.COCO),
    ("bdd100k", Dataset.BDD100K_TRAIN, Dataset.BDD100k_VAL, Evaluator.BDD100K),
    ("coco", Dataset.COCO_TRAIN, Dataset.COCO_VAL, Evaluator.COCO),
    ("lvis", Dataset.LVIS_TRAIN, Dataset.LVIS_MINIVAL, Evaluator.LVIS),
]


DATASETS_AND_Extraction: List[Tuple[str, Dataset, Dataset, Evaluator]] = [
    (
        "funnybirds",
        Dataset.FUNNYBIRDS_NO_INTERVENTION,
        Dataset.FUNNYBIRDS_NO_BEAK,
        Evaluator.COCO,
    ),
    ("valerie", Dataset.VALERIE22_TRAIN, Dataset.VALERIE22_VAL, Evaluator.COCO),
    ("ecpersons", Dataset.ECPERSONS_TRAIN, Dataset.ECPERSONS_VAL, Evaluator.COCO),
    ("kitti", Dataset.KITTI_TRAIN, Dataset.KITTI_VAL, Evaluator.COCO),
    ("bdd100k", Dataset.BDD100K_TRAIN, Dataset.BDD100k_VAL, Evaluator.BDD100K),
    ("coco", Dataset.COCO_TRAIN, Dataset.COCO_VAL, Evaluator.COCO),
    ("lvis", Dataset.LVIS_TRAIN, Dataset.LVIS_MINIVAL, Evaluator.LVIS),
]
