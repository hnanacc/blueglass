# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random
from glob import glob
from typing import Tuple, List, Dict

DATASET_DIR = os.environ.get("DATASET_DIR")

FUNNYBIRDS_CUSTOM_ROOT = f"{DATASET_DIR}/funnybirds/custom"
FUNNYBIRDS_COCO_IMAGES = f"{DATASET_DIR}/funnybirds/cocofied/images"
FUNNYBIRDS_COCO_ANNOTS = f"{DATASET_DIR}/funnybirds/cocofied/annotations"

BACKGROUND_COLOR = (153, 85, 85)  # BGR
IMAGE_SIZE = (1024, 1024, 3)  # H, W, C
GRID_DIM = 256  # DEPENDS ON GENERATED IMAGE SIZE.
NUM_OBJ_PER_IMAGE = 5  # some
NUM_SAMPLES = 10  # NUMBER OF GENERATED IMAGES.


def prepare_trimmed_crop_with_bbox(image):
    pass


def transform_crops(crop, bbox: Tuple[int, int, int, int]):
    pass


def sample_random_centers():
    cx = np.arange(GRID_DIM // 2, IMAGE_SIZE[1] - (GRID_DIM // 2), GRID_DIM)
    cy = np.arange(GRID_DIM // 2, IMAGE_SIZE[0] - (GRID_DIM // 2), GRID_DIM)

    vx, vy = np.meshgrid(cx, cy)

    return random.sample(
        [(xi, yi) for xi, yi in zip(vx.flatten(), vy.flatten())], k=NUM_OBJ_PER_IMAGE
    )


CROP_SOURCE_DIR = f"{FUNNYBIRDS_CUSTOM_ROOT}/isometric_camera"

SPLITS = (
    "isometric_camera",
    "isometric_no_beak",
    "isometric_no_eyes",
    "isometric_no_foot",
    "isometric_no_tail",
    "isometric_no_wings",
)


def trim_crop(crop_path: str):
    crop = cv2.imread(crop_path)

    nrows, ncols = crop.shape[:2]

    is_background_row = (crop == BACKGROUND_COLOR).all(axis=0)
    is_background_col = (crop == BACKGROUND_COLOR).all(axis=1)

    xmin, xmax, ymin, ymax = 0, 0, 0, 0

    for r in range(nrows):
        if not is_background_row[r].all():
            break
        xmin = r

    for r in range(nrows - 1, -1, -1):
        if not is_background_row[r].all():
            break
        xmax = r

    for c in range(ncols):
        if not is_background_col[c].all():
            break
        ymin = c

    for c in range(ncols - 1, -1, -1):
        if not is_background_col[c].all():
            break
        ymax = c

    return np.ascontiguousarray(crop[ymin:ymax, xmin:xmax, :])


def prepare_image_from_crops_and_centers(
    crop_paths: List[str], centers: List[Tuple[int, int]]
):
    canvas = np.ones(IMAGE_SIZE, dtype=np.uint8)
    canvas = canvas * BACKGROUND_COLOR

    bboxes = []
    for crop_path, center in zip(crop_paths, centers):
        crop = trim_crop(crop_path)
        h, w = crop.shape[:2]

        if h % 2 != 0:
            crop = crop[1:, :, :]

        if w % 2 != 0:
            crop = crop[:, 1:, :]

        xmin = center[0] - (w // 2)
        xmax = center[0] + (w // 2)
        ymin = center[1] - (h // 2)
        ymax = center[1] + (h // 2)

        canvas[ymin:ymax, xmin:xmax, :] = crop

        bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

    return np.ascontiguousarray(canvas), bboxes


def to_coco_image(imid: str, split: str):
    return {
        "id": imid,
        "file_name": f"{split}/{imid}.png",
        "height": IMAGE_SIZE[0],
        "width": IMAGE_SIZE[1],
    }


def to_coco_annos(imid: str, bboxes: List[Tuple[int, int, int, int]], clsid: int):
    return [
        {
            "id": f"{imid}_{idx:03}",
            "image_id": imid,
            "bbox": [int(i) for i in bb],
            "category_id": clsid,
            "ignore": 0,
            "area": int(bb[2]) * int(bb[3]),
            "iscrowd": 0,
            "truncated": False,
        }
        for idx, bb in enumerate(bboxes)
    ]


def save_image(im_path: str, im):
    dirname = os.path.dirname(im_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cv2.imwrite(im_path, im)


def separate_imid_and_identifier(im_dicts: List[Dict], an_dicts: List[Dict]):
    # create a map of old_image_ids to new_ids
    map_old_to_new_imid = {}

    for new_imid, im_dict in enumerate(im_dicts):
        map_old_to_new_imid[im_dict["id"]] = new_imid
        im_dict["identifier"] = im_dict["id"]
        im_dict["id"] = new_imid

    for new_anid, an_dict in enumerate(an_dicts):
        an_dict["image_id"] = map_old_to_new_imid[an_dict["image_id"]]
        an_dict["identifier"] = an_dict["id"]
        an_dict["id"] = new_anid

    return im_dicts, an_dicts


def prepare_funnybirds_coco_annotations():
    if not os.path.exists(FUNNYBIRDS_COCO_ANNOTS):
        os.makedirs(FUNNYBIRDS_COCO_ANNOTS)

    if not os.path.exists(FUNNYBIRDS_COCO_IMAGES):
        os.makedirs(FUNNYBIRDS_COCO_IMAGES)

    images, annots = defaultdict(list), defaultdict(list)
    crop_source = list(sorted(glob(f"{CROP_SOURCE_DIR}/*.png")))

    for idx in tqdm(range(NUM_SAMPLES), "processing samples"):
        imid = f"{idx:05}"

        chosens = random.sample(crop_source, k=NUM_OBJ_PER_IMAGE)
        centers = sample_random_centers()

        for split in SPLITS:
            chosens_paths = [
                f"{FUNNYBIRDS_CUSTOM_ROOT}/{split}/{os.path.basename(p)}"
                for p in chosens
            ]
            im, bb = prepare_image_from_crops_and_centers(chosens_paths, centers)
            save_image(f"{FUNNYBIRDS_COCO_IMAGES}/{split}/{imid}.png", im)
            images[split].append(to_coco_image(imid, split))
            annots[split].extend(to_coco_annos(imid, bb, 0))

    for split in tqdm(SPLITS, desc="saving cocofied annotations"):
        im, an = images[split], annots[split]
        im, an = separate_imid_and_identifier(im, an)

        coco_dict = {
            "info": {"description": "FUNNYBIRDS Custom", "contributor": "xVLM"},
            "categories": [
                {"supercategory": "bird", "id": 0, "name": "bird", "split": split},
            ],
            "images": im,
            "annotations": an,
        }
        anno_path = f"{FUNNYBIRDS_COCO_ANNOTS}/instances_{split}.json"
        with open(anno_path, "w") as fp:
            json.dump(coco_dict, fp)


if __name__ == "__main__":
    prepare_funnybirds_coco_annotations()
