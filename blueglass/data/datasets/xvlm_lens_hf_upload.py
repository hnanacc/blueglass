# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from huggingface_hub import HfApi

repo_id = "IntelLabs/BLUE-Lens"
dataset_path = "BLUE-Lens/features_datasets/gdino/gdino.coco_mini"

api = HfApi()
api.upload_folder(
    folder_path=dataset_path,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="features_datasets/gdino/gdino.coco_mini",
)

print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
