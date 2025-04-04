# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import json
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from glob import glob
import logging
from typing import List, Dict, Union

DATASET_DIR = os.environ.get("DATASET_DIR")

VALERIE22_ROOT = f"{DATASET_DIR}/VALERIE22/data"
VALERIE22_COCOFIED_LABEL_PATH = "cocofied_annotations/instances_{}.json"
VALERIE22_SEQUENCE_LABEL_PATH = "{}/ground-truth/2d-bounding-box_json/{}.json"
VALERIE22_SEQUENCE_IMAGE_PATH = "{}/sensor/camera/left/png/{}.png"

SPLITS = {
    "train": [
        "intel_results_sequence_0057",
        "intel_results_sequence_0058",
        "intel_results_sequence_0059",
        "intel_results_sequence_0060",
    ],
    "val": ["intel_results_sequence_0062", "intel_results_sequence_0062_b"],
    "mini": ["intel_results_sequence_0057"],
}

logger = logging.getLogger("parser")


def parse_bbox_from_valerie(bbox: Dict[str, Union[float, int]]):
    cx, cy, h, w = bbox["c_x"], bbox["c_y"], bbox["h"], bbox["w"]
    xmin, ymin = cx - (w / 2), cy - (h / 2)
    return [xmin, ymin, w, h]


def convert_ids_to_coco_instances(split_sample_ids: List[str]):
    images, annotations = [], []

    for seq, si in tqdm(split_sample_ids, desc="parse ids to coco instances"):
        fanno = os.path.join(
            VALERIE22_ROOT, VALERIE22_SEQUENCE_LABEL_PATH.format(seq, si)
        )
        with open(fanno, "r") as fp:
            annos = json.load(fp)

        images.append(
            {
                "id": si,
                "file_name": VALERIE22_SEQUENCE_IMAGE_PATH.format(seq, si),
                "height": 1200,
                "width": 1920,
            }
        )

        annotations.extend(
            [
                {
                    "id": f"{aid}_{si}",
                    "image_id": si,
                    "bbox": parse_bbox_from_valerie(ano["bb"]),
                    "bbox_vis": parse_bbox_from_valerie(ano["bb_vis"]),
                    "category_id": 1,
                    "truncated": ano["truncated"],
                    "is_crowd": 0,
                }
                for aid, ano in annos.items()
                if "bb" in ano and "bb_vis" in ano
            ]
        )

    return {
        "info": {
            "description": "VALERIE22 cocofied",
            "contributor": "xVLM",
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"supercategory": "person", "id": 1, "name": "person"}],
    }


def main():
    for split, sequences in tqdm(SPLITS.items(), desc="parse each split"):
        split_sample_ids = []

        for seq in tqdm(sequences, desc="parse each sequence"):
            im_paths = glob(
                os.path.join(
                    VALERIE22_ROOT, VALERIE22_SEQUENCE_IMAGE_PATH.format(seq, "*")
                )
            )
            im_smids = [os.path.basename(p).rstrip(".png") for p in im_paths]
            sq_smids = [
                (seq, i)
                for i in im_smids
                if os.path.exists(
                    os.path.join(
                        VALERIE22_ROOT, VALERIE22_SEQUENCE_LABEL_PATH.format(seq, i)
                    )
                )
            ]
            logger.info(
                f"Found {len(sq_smids)} valid / {len(im_paths)} total samples in seq: {seq}."
            )
            split_sample_ids.extend(sq_smids)

        logger.info(f"Found {len(split_sample_ids)} samples in split: {split}.")
        instances = convert_ids_to_coco_instances(sorted(split_sample_ids))

        coco_path = os.path.join(
            VALERIE22_ROOT, VALERIE22_COCOFIED_LABEL_PATH.format(split)
        )

        if not os.path.exists(os.path.dirname(coco_path)):
            os.makedirs(os.path.dirname(coco_path))

        with open(coco_path, "w") as fp:
            json.dump(instances, fp)

        logger.info(f"saved split: {split} file at {coco_path}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()
