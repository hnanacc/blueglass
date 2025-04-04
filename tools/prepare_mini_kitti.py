# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
import logging
import random
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DATASET_DIR = os.environ.get("DATASET_DIR")

random.seed(1337)
logger = logging.getLogger("parser")

KITTI_ANNO_ROOT = f"{DATASET_DIR}/KITTI/cocofied/annotations"

KITTI_SOURCE_ANNO_FILE = os.path.join(KITTI_ANNO_ROOT, "instances_val.json")
KITTI_DESTIN_ANNO_FILE = os.path.join(KITTI_ANNO_ROOT, "instances_mini.json")

MINI_SIZE = 100


def main():
    assert os.path.exists(KITTI_SOURCE_ANNO_FILE), "source annotations do not exists."

    if os.path.exists(KITTI_DESTIN_ANNO_FILE):
        logger.info("Mini anno file already exists, overriding...")

    with open(KITTI_SOURCE_ANNO_FILE, "r") as fp:
        src_annos = json.load(fp)

    images = random.sample(src_annos["images"], MINI_SIZE)
    im_ids = [im["id"] for im in images]
    annots = [an for an in tqdm(src_annos["annotations"]) if an["image_id"] in im_ids]

    dst_annos = {
        **src_annos,
        "images": images,
        "annotations": annots,
    }

    with open(KITTI_DESTIN_ANNO_FILE, "w") as fp:
        json.dump(dst_annos, fp)

    logger.info(f"Saved new mini annos at: {KITTI_DESTIN_ANNO_FILE}")
    logger.info("Finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        main()
