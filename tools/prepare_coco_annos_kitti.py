# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import glob
import json
import shutil
import random
from PIL import Image
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List, Dict, Tuple, Any

DATASET_DIR = os.environ.get("DATASET_DIR")

random.seed(1337)

KITTI_ROOT = f"{DATASET_DIR}/KITTI/2d_object"
ORIG_ANNOS_PATH = os.path.join(KITTI_ROOT, "training/label_2")
ORIG_IMAGE_PATH = os.path.join(KITTI_ROOT, "training/image_2")

COCO_ROOT = f"{DATASET_DIR}/KITTI/cocofied"
COCO_IMAGE_PATH = os.path.join(COCO_ROOT, "images")
COCO_ANNOS_PATH = os.path.join(COCO_ROOT, "annotations")

TRAIN_SPLIT_RATIO = 0.8

KITTI_CATEGORIES = ["car", "pedestrian", "cyclist"]
IGNORE_CLASSES = ["van", "truck", "person_sitting", "tram", "misc", "dontcare"]

logger = logging.getLogger("parser")


def kitti_line_to_dict(line: str) -> Dict[str, Any]:
    f = line.strip().split(" ")
    assert len(f) == 15, "unexpected line, has invalid fields."
    return {
        "type": f[0].lower(),
        "truncated": float(f[1]),  # float: 0-1, 1=fully truncated.
        "occluded": int(f[2]),  # 0,1,2,3 3=fully occuluded.
        "alpha": float(f[3]),  # -pi to pi.
        "bbox": list(map(float, [f[4], f[5], f[6], f[7]])),
        "dimensions": list(map(float, [f[8], f[9], f[10]])),  # 3d dims, h, w, l.
        "location": list(map(float, [f[11], f[12], f[13]])),  # 3d posi, camera coords.
        "rotation": float(f[14]),  # rotation around y-axis, -pi to pi.
    }


def parse_kitti(path: str) -> Dict[str, Any]:
    with open(path, "r") as fp:
        annos = [kitti_line_to_dict(line) for line in fp]

    image_id = os.path.basename(path).removesuffix(".txt")

    return {
        "image_id": image_id,
        "file_name": os.path.join(COCO_IMAGE_PATH, f"{image_id}.png"),
        "annotations": annos,
    }


def parse_coco_fields(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    images, labels = [], []

    for smp in tqdm(samples, desc="parse coco fields"):
        # check if samples is valid.
        image_id = smp["image_id"]
        has_image = os.path.exists(os.path.join(ORIG_IMAGE_PATH, f"{image_id}.png"))
        has_label = os.path.exists(os.path.join(ORIG_ANNOS_PATH, f"{image_id}.txt"))

        if not has_label or not has_image:
            continue

        # compute image height and width.
        im_w, im_h = Image.open(os.path.join(COCO_IMAGE_PATH, smp["file_name"])).size

        images.append(
            {
                "id": smp["image_id"],
                "file_name": smp["file_name"],
                "height": im_h,
                "width": im_w,
            }
        )

        for anno in smp["annotations"]:
            if anno["type"] not in KITTI_CATEGORIES:
                continue

            # compute box coordinates in coco format.
            x1, y1, x2, y2 = anno["bbox"]

            assert x2 > x1 and y2 > y1, "invalid coordinates."

            labels.append(
                {
                    "category_id": KITTI_CATEGORIES.index(anno["type"]),
                    "iscrowd": 0,
                    "image_id": smp["image_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )

    labels = [{**lab, "id": aid} for aid, lab in enumerate(labels)]

    return images, labels


def confirm_path_deletion(path: str):
    c = input("coco annotations already exists, continue? (y/N): ")
    if c.lower() != "y":
        exit(0)


def main():
    # prepare coco folder structure.
    if os.path.exists(COCO_ROOT):
        confirm_path_deletion(COCO_ROOT)
        shutil.rmtree(COCO_ROOT)

    os.makedirs(COCO_ROOT)
    os.makedirs(COCO_IMAGE_PATH)
    os.makedirs(COCO_ANNOS_PATH)

    # load images.
    samples = glob.glob(os.path.join(ORIG_IMAGE_PATH, f"*.png"))

    for smp in tqdm(samples, desc="copying images to coco path"):
        shutil.copy(smp, COCO_IMAGE_PATH)

    # load annotations.
    samples = list(sorted(glob.glob(os.path.join(ORIG_ANNOS_PATH, f"*.txt"))))

    n_pp_samples = len(samples)
    n_tr_samples = int(n_pp_samples * TRAIN_SPLIT_RATIO)
    sample_tr_id = random.sample(range(0, n_pp_samples), n_tr_samples)
    sample_vd_id = [idn for idn in range(0, n_pp_samples) if idn not in sample_tr_id]

    sample_train = [
        parse_kitti(samples[sid])
        for sid in tqdm(sample_tr_id, desc="parse train samples")
    ]
    sample_valid = [
        parse_kitti(samples[sid])
        for sid in tqdm(sample_vd_id, desc="parse valid samples")
    ]

    train_images, train_labels = parse_coco_fields(sample_train)
    valid_images, valid_labels = parse_coco_fields(sample_valid)

    base_coco = {
        "info": {
            "author": "BLUEGLASS",
            "contributor": "BLUEGLASS",
            "description": "KITTI 2D Cocofied Annotations",
        },
        "licences": [{"id": 0, "name": "MIT", "url": "https://mit-license.org/"}],
        "categories": [
            {"supercategory": name, "id": idx, "name": name}
            for idx, name in enumerate(KITTI_CATEGORIES)
        ],
    }

    train_coco = {
        **base_coco,
        "images": train_images,
        "annotations": train_labels,
    }

    valid_coco = {
        **base_coco,
        "images": valid_images,
        "annotations": valid_labels,
    }

    train_path = os.path.join(COCO_ANNOS_PATH, "instances_train.json")
    valid_path = os.path.join(COCO_ANNOS_PATH, "instances_val.json")

    with open(train_path, "w") as fp:
        json.dump(train_coco, fp)
    logger.info(f"saved train annotations to: {train_path}")

    with open(valid_path, "w") as fp:
        json.dump(valid_coco, fp)
    logger.info(f"saved valid annotations to: {valid_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()
