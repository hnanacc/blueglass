# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

"""
COCO annotations for Open Images dataset.
"""

import os
import json
import random
import cv2
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List, Dict, Tuple, Any
import pandas as pd

DATASET_DIR = os.environ.get("DATASET_DIR")

random.seed(1337)

OI_ROOT = f"{DATASET_DIR}/openimages"

OI_ANNOS_TMPL = os.path.join(OI_ROOT, "annotations", "oi6-{}-annotations-bbox.csv")
OI_IMAGE_TMPL = os.path.join(OI_ROOT, "images", "{}", "{}.jpg")
OI_CLASSNAMES = os.path.join(OI_ROOT, "metadata", "oi6-classnames.csv")

OI_COCO_TMPL = os.path.join(OI_ROOT, "annotations/cocofied_oi6_{}.json")

SPLITS = ["validation", "train"]


logger = logging.getLogger("parser")


def prepare_images(
    split: str, annos: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Tuple[float, float]]]:
    im_ids = annos.ImageID.unique().tolist()

    images, imid_to_ind, imid_sizes = [], {}, {}
    ind_counter = 1

    for im_id in tqdm(im_ids, desc="prepare images"):
        im_path = OI_IMAGE_TMPL.format(split, im_id)
        if not os.path.exists(im_path):
            continue

        im_h, im_w = cv2.imread(im_path).shape[:2]

        images.append(
            {
                "id": ind_counter,
                "imid": im_id,
                "file_name": os.path.basename(im_path),
                "height": im_h,
                "width": im_w,
            }
        )
        imid_to_ind[im_id] = ind_counter
        imid_sizes[im_id] = (im_h, im_w)

        ind_counter += 1

    return images, imid_to_ind, imid_sizes


def prepare_categories():
    cnm = pd.read_csv(OI_CLASSNAMES)
    cnames, cmid_to_ind = [], {}
    ind_counter = 1

    for mid, dnm in zip(cnm.LabelName, cnm.DisplayName):
        dnm = "_".join(stem.strip() for stem in dnm.lower().split())

        # TODO: replace superctegory with openimages
        #       provided relationships.

        cnames.append({"supercategory": dnm, "id": ind_counter, "name": dnm})
        cmid_to_ind[mid] = ind_counter

        ind_counter += 1

    return cnames, cmid_to_ind


def bbox_oi_to_coco(anno, imid_sizes):
    im_h, im_w = imid_sizes[anno.ImageID]
    xmin = float(anno.XMin) * im_w
    ymin = float(anno.YMin) * im_h
    xmax = float(anno.XMax) * im_w
    ymax = float(anno.YMax) * im_h

    return [xmin, ymin, (xmax - xmin), (ymax - ymin)]


def prepare_labels(
    annos: pd.DataFrame,
    imid_to_ind: Dict[str, int],
    cmid_to_ind: Dict[str, int],
    imid_sizes: Dict[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:

    return [
        {
            "id": ind,
            "category_id": cmid_to_ind[anno.LabelName],
            "image_id": imid_to_ind[anno.ImageID],
            "bbox": bbox_oi_to_coco(anno, imid_sizes),
            "iscrowd": anno.IsGroupOf,
        }
        for ind, anno in tqdm(annos.iterrows(), desc="prepare annos")
    ]


def main():
    for split in SPLITS:
        logger.info(f"Process {split} set.")

        annos = pd.read_csv(OI_ANNOS_TMPL.format(split))

        cnames, cmid_to_ind = prepare_categories()
        images, imid_to_ind, imid_sizes = prepare_images(split, annos)
        labels = prepare_labels(annos, imid_to_ind, cmid_to_ind, imid_sizes)

        coco_annos = {
            "info": {
                "author": "BLUEGLASS",
                "contributor": "BLUEGLASS",
                "description": "OpenImagesv6 Cocofied Annotations",
            },
            "licences": [{"id": 0, "name": "MIT", "url": "https://mit-license.org/"}],
            "categories": cnames,
            "images": images,
            "annotations": labels,
        }

        path = OI_COCO_TMPL.format(split)

        with open(path, "w") as fp:
            json.dump(coco_annos, fp)

        logger.info(f"Saved cocofied {split} annotations to: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()
