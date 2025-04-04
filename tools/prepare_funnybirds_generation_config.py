# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import math
import os
import random
import json
from typing import List, Dict

N_SAMPLES = 100

random.seed(1337)

DATASET_DIR = os.environ.get("DATASET_DIR")

FUNNYBIRDS_ROOT = f"{DATASET_DIR}/funnybirds"
FUNNYBIRDS_CUSTOM_ROOT = FUNNYBIRDS_ROOT + "/custom"
FUNNYBIRDS_PART_CONFIG = FUNNYBIRDS_ROOT + "/original/parts.json"


def sample_isometric_camera_position():
    """
    Because the data is to analyze concept,
    we fix the camera in a position such that
    all the concept are visible in the image.
    """
    return {
        "camera_distance": random.randint(200, 800),
        "camera_pitch": 5.5,
        "camera_roll": 4.0,
        "light_distance": 300,
        "light_pitch": random.uniform(0.0, math.pi * 2),
        "light_roll": random.uniform(0.0, math.pi * 2),
    }


def sample_front_camera_position():
    """
    Because the data is to analyze concept,
    we fix the camera in a position such that
    all the concept are visible in the image.
    """
    return {
        "camera_distance": random.randint(200, 700),
        "camera_pitch": 3.2,
        "camera_roll": 10.5,
        "light_distance": 300,
        "light_pitch": random.uniform(0.0, math.pi * 2),
        "light_roll": random.uniform(0.0, math.pi * 2),
    }


def sample_random_camera_position():
    """
    In case we want to analyze with random camera angles.
    Will be parts be activated even when it's not visible?
    """
    return {
        "camera_distance": random.randint(200, 700),
        "camera_pitch": random.uniform(0.0, math.pi * 2),
        "camera_roll": random.uniform(0.0, math.pi * 2),
        "light_distance": 300,
        "light_pitch": random.uniform(0.0, math.pi * 2),
        "light_roll": random.uniform(0.0, math.pi * 2),
    }


def sample_parts_dict(parts_spec: Dict[str, List[Dict[str, str]]]):
    beak = random.choice(parts_spec["beak"])
    eyes = random.choice(parts_spec["eye"])
    foot = random.choice(parts_spec["foot"])
    tail = random.choice(parts_spec["tail"])
    wing = random.choice(parts_spec["wing"])

    return {
        "beak_model": beak["model"],
        "beak_color": beak["color"],
        "eye_model": eyes["model"],
        "foot_model": foot["model"],
        "tail_model": tail["model"],
        "tail_color": tail["color"],
        "wing_model": wing["model"],
        "wing_color": wing["color"],
    }


def no_background_sample():
    return {
        "bg_objects": "",
        "bg_radius": "",
        "bg_pitch": "",
        "bg_roll": "",
        "bg_scale_x": "",
        "bg_scale_y": "",
        "bg_scale_z": "",
        "bg_rot_x": "",
        "bg_rot_y": "",
        "bg_rot_z": "",
        "bg_color": "",
    }


def save_json(file, fname: str):
    with open(os.path.join(FUNNYBIRDS_CUSTOM_ROOT, fname), "w") as fp:
        json.dump(file, fp)
    print(f"Saved json to {fname}")


def prepare_funnybirds_generation_config(parts_spec):
    parts_samples = [sample_parts_dict(parts_spec) for _ in range(N_SAMPLES)]

    # add backgrounds to all dicts.
    parts_samples = [{**parts, **no_background_sample()} for parts in parts_samples]

    # generate with isometric camera view.
    isometric_camera_dicts = [
        {**sample_isometric_camera_position(), **parts} for parts in parts_samples
    ]

    frontal_camera_dicts = [
        {**sample_front_camera_position(), **parts} for parts in parts_samples
    ]

    save_json(isometric_camera_dicts, "conf_isometric_camera.json")
    save_json(frontal_camera_dicts, "conf_frontal_camera.json")
    save_json(isometric_camera_dicts + frontal_camera_dicts, "conf_mixed_camera.json")

    save_json(
        [
            {**dicts, "beak_model": "", "beak_color": ""}
            for dicts in isometric_camera_dicts
        ],
        "conf_isometric_no_beak.json",
    )

    save_json(
        [{**dicts, "eye_model": ""} for dicts in isometric_camera_dicts],
        "conf_isometric_no_eyes.json",
    )

    save_json(
        [
            {**dicts, "foot_model": "", "foot_color": ""}
            for dicts in isometric_camera_dicts
        ],
        "conf_isometric_no_foot.json",
    )

    save_json(
        [
            {**dicts, "tail_model": "", "tail_color": ""}
            for dicts in isometric_camera_dicts
        ],
        "conf_isometric_no_tail.json",
    )

    save_json(
        [
            {**dicts, "wing_model": "", "wing_color": ""}
            for dicts in isometric_camera_dicts
        ],
        "conf_isometric_no_wings.json",
    )


if __name__ == "__main__":
    assert os.path.exists(FUNNYBIRDS_PART_CONFIG), "parts.json doesn't exists."
    with open(FUNNYBIRDS_PART_CONFIG, "r") as fp:
        parts_spec = json.load(fp)

    if not os.path.exists(FUNNYBIRDS_CUSTOM_ROOT):
        os.makedirs(FUNNYBIRDS_CUSTOM_ROOT)

    prepare_funnybirds_generation_config(parts_spec)
