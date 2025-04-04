# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
import logging
import random
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DATASET_DIR = os.environ.get("DATASET_DIR")

random.seed(1337)
logger = logging.getLogger("parser")

OI_ANNO_ROOT = f"{DATASET_DIR}/openimages/annotations"

OI_MINI_SOURCE_ANNO_FILE = os.path.join(OI_ANNO_ROOT, "cocofied_oi6_validation.json")
OI_MINI_DESTIN_ANNO_FILE = os.path.join(OI_ANNO_ROOT, "cocofied_oi6_mini.json")

OI_SHARD_SOURCE_ANNO_FILE = os.path.join(OI_ANNO_ROOT, "cocofied_oi6_train.json")
OI_SHARD_DESTIN_ANNO_TMPL = os.path.join(OI_ANNO_ROOT, "cocofied_oi6_shard_{}.json")

SHARDS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
]

MINI_SIZE = 100


def prepare_mini():
    assert os.path.exists(
        OI_MINI_SOURCE_ANNO_FILE
    ), "mini source annotation doesn't exist."

    with open(OI_MINI_SOURCE_ANNO_FILE, "r") as fp:
        src_annos = json.load(fp)

    if os.path.exists(OI_MINI_DESTIN_ANNO_FILE):
        logger.info("Found existing mini annotation file, overwriting...")

    images = random.sample(src_annos["images"], MINI_SIZE)
    im_ids = set([im["id"] for im in images])
    annots = [
        an
        for an in tqdm(src_annos["annotations"], desc="mini annos")
        if an["image_id"] in im_ids
    ]

    dst_annos = {
        **src_annos,
        "images": images,
        "annotations": annots,
    }

    with open(OI_MINI_DESTIN_ANNO_FILE, "w") as fp:
        json.dump(dst_annos, fp)

    logger.info(f"Saved mini annos at: {OI_MINI_DESTIN_ANNO_FILE}")


def prepare_shards():
    assert os.path.exists(
        OI_SHARD_SOURCE_ANNO_FILE
    ), "shard source annotation doesn't exist."

    logger.info("Loading annotations file...")

    with open(OI_SHARD_SOURCE_ANNO_FILE, "r") as fp:
        src_annos = json.load(fp)

    images_per_shard = defaultdict(list)

    for im in tqdm(src_annos["images"], desc="group images"):
        images_per_shard[im["imid"][0]].append(im)

    assert len(images_per_shard) == len(
        SHARDS
    ), "unexpected: there are more image groups than shards."

    num_parsed_annos = 0

    for shard in SHARDS:
        im_ids = set([im["id"] for im in images_per_shard[shard]])
        annots = [
            an
            for an in tqdm(src_annos["annotations"], desc=f"shard {shard} annos")
            if an["image_id"] in im_ids
        ]

        num_parsed_annos += len(annots)

        dst_annos = {
            **src_annos,
            "images": images_per_shard[shard],
            "annotations": annots,
        }

        path = OI_SHARD_DESTIN_ANNO_TMPL.format(shard)

        with open(path, "w") as fp:
            json.dump(dst_annos, fp)

        logger.info(f"Saved shard {shard} annos at: {path}")

    assert num_parsed_annos == len(
        src_annos["annotations"]
    ), "unexpected: num annotations in shard and source mismatch."


def main():
    prepare_mini()
    prepare_shards()

    logger.info("Finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        main()
