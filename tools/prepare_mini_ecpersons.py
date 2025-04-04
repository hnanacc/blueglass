# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
from shutil import copyfile
import logging
import random
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DATASET_DIR = os.environ.get("DATASET_DIR")

random.seed(1337)
logger = logging.getLogger("parser")

ECP_ROOT_PATH = f"{DATASET_DIR}/eurocity"

ECP_SOURCE_ANNO_FILE = os.path.join(
    ECP_ROOT_PATH, "cocofied_annotations/instances_day_val.json"
)

ECP_DESTIN_ANNO_FILE = os.path.join(
    ECP_ROOT_PATH, "cocofied_annotations/instances_day_mini.json"
)

ECP_GT_LABELS_PATH = os.path.join(ECP_ROOT_PATH, "ECP/day/labels/mini")

MINI_SIZE = 300


def main():
    if os.path.exists(ECP_DESTIN_ANNO_FILE):
        logger.info("Mini anno file already exists, overriding...")

    with open(ECP_SOURCE_ANNO_FILE, "r") as fp:
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

    with open(ECP_DESTIN_ANNO_FILE, "w") as fp:
        json.dump(dst_annos, fp)

    logger.info(f"Saved new mini annos at: {ECP_DESTIN_ANNO_FILE}")
    logger.info("Copying images to new mini folder.")

    if not os.path.exists(ECP_GT_LABELS_PATH):
        os.makedirs(ECP_GT_LABELS_PATH)

    for im in tqdm(d_ims):
        src_fn = os.path.join(ECP_ROOT_PATH, im["file_name"])
        src_fn = src_fn.replace("img", "labels")
        src_fn = src_fn.replace("png", "json")

        dst_fn = os.path.join(ECP_GT_LABELS_PATH, os.path.basename(im["file_name"]))
        dst_fn = dst_fn.replace("png", "json")

        copyfile(src_fn, dst_fn)

    logger.info("Finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        main()
