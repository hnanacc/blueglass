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

LVIS_ROOT_PATH = f"{DATASET_DIR}/lvis"
LVIS_SOURCE_ANNO_FILE = os.path.join(LVIS_ROOT_PATH, "lvis_v1_val.json")
LVIS_DESTIN_ANNO_FILE = os.path.join(LVIS_ROOT_PATH, "lvis_v1_mini.json")

MINI_SIZE = 300


def main():
    if os.path.exists(LVIS_DESTIN_ANNO_FILE):
        logger.info("Mini anno file already exists, overriding...")

    with open(LVIS_SOURCE_ANNO_FILE, "r") as fp:
        src_annos = json.load(fp)

    s_ims = src_annos["images"]
    d_ims = [s_ims[random.randint(0, len(s_ims) - 1)] for _ in tqdm(range(MINI_SIZE))]
    d_ids = [im["id"] for im in d_ims]
    d_ano = [anno for anno in src_annos["annotations"] if anno["image_id"] in d_ids]

    dst_annos = {
        "info": src_annos["info"],
        "licenses": src_annos["licenses"],
        "categories": src_annos["categories"],
        "images": d_ims,
        "annotations": d_ano,
    }

    with open(LVIS_DESTIN_ANNO_FILE, "w") as fp:
        json.dump(dst_annos, fp)

    logger.info(f"Saved new mini annos at: {LVIS_DESTIN_ANNO_FILE}")
    logger.info("Finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        main()
