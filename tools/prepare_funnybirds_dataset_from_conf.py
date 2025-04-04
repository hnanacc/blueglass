# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import random
import requests
from PIL import Image
from base64 import decodebytes
import io
import os
from glob import glob
from shutil import rmtree
import json
from tqdm import tqdm

DATASET_DIR = os.environ.get("DATASET_DIR")

FUNNYBIRDS_CUSTOM_ROOT = f"{DATASET_DIR}/funnybirds/custom"

random.seed(1337)


def dict_to_url(dict, prefix="http://localhost:8082/render?", render_mode="default"):
    dict["render_mode"] = render_mode
    url = prefix + "&".join([f"{k}={v}" for k, v in dict.items()]).replace("=&", '=""&')
    print(url)
    return url


def dict_to_image(dict):
    return Image.open(
        io.BytesIO(decodebytes(requests.get(dict_to_url(dict)).content))
    ).resize((256, 256))


def prepare_funnybirds_dataset_from_conf(conf_paths):
    for conf_path in tqdm(conf_paths, desc="per conf"):
        im_path = os.path.join(
            FUNNYBIRDS_CUSTOM_ROOT,
            os.path.basename(conf_path).replace("conf_", "").replace(".json", ""),
        )

        if os.path.exists(im_path):
            rmtree(im_path)
        os.makedirs(im_path)

        with open(conf_path, "r") as fp:
            conf = json.load(fp)

        for idx, sample in enumerate(tqdm(conf, desc="per sample")):
            dict_to_image(sample).save(os.path.join(im_path, f"{idx:05}.png"), "png")


def cleanup():
    pattern = os.path.join("/tmp", "puppeteer*")
    for item in glob(pattern):
        if not os.path.isdir(item):
            continue
        rmtree(item)


if __name__ == "__main__":
    conf_paths = list(sorted(glob(FUNNYBIRDS_CUSTOM_ROOT + "/conf*.json")))
    prepare_funnybirds_dataset_from_conf(conf_paths)
