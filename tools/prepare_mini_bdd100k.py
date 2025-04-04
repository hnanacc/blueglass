# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
import logging
import random
from shutil import copyfile
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DATASET_DIR = os.environ.get("DATASET_DIR")

random.seed(1337)
logger = logging.getLogger("parser")

BDD100K_SOURCE_ANNO_FILE = "datasets/bdd100k/labels/det_20/det_val.json"
BDD100K_SOURCE_IMGS_PATH = f"{DATASET_DIR}/bdd100k/images/100k/val"
BDD100K_DESTIN_ANNO_FILE = f"{DATASET_DIR}/bdd100k/labels/det_20/det_mini.json"
BDD100K_DESTIN_IMGS_PATH = f"{DATASET_DIR}/bdd100k/images/100k/mini"

MINI_SIZE = 300


def main():
    if os.path.exists(BDD100K_DESTIN_ANNO_FILE):
        logger.info("Mini anno file already exists, overriding...")

    with open(BDD100K_SOURCE_ANNO_FILE, "r") as fp:
        src_annos = json.load(fp)

    dst_annos = [
        src_annos[random.randint(0, len(src_annos) - 1)] for _ in tqdm(range(MINI_SIZE))
    ]

    with open(BDD100K_DESTIN_ANNO_FILE, "w") as fp:
        json.dump(dst_annos, fp)

    logger.info(f"Saved new mini annos at: {BDD100K_DESTIN_ANNO_FILE}")
    logger.info(f"Copying images to new mini folder: {BDD100K_DESTIN_IMGS_PATH}")

    if not os.path.exists(BDD100K_DESTIN_IMGS_PATH):
        os.makedirs(BDD100K_DESTIN_IMGS_PATH)

    ims_names = [ano["name"] for ano in dst_annos]
    for nm in tqdm(ims_names):
        copyfile(
            os.path.join(BDD100K_SOURCE_IMGS_PATH, nm),
            os.path.join(BDD100K_DESTIN_IMGS_PATH, nm),
        )

    logger.info("Finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        main()
