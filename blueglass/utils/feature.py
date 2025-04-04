# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
import shutil
from blueglass.utils.logger_utils import setup_blueglass_logger
from huggingface_hub import (
    list_repo_files,
    hf_hub_download,
    get_paths_info,
    snapshot_download,
)
from blueglass.configs import BLUEGLASSConf, Dataset, Model

from typing import List, Dict, Any, Optional

logger = setup_blueglass_logger(__name__)

REPO_ID = "IntelLabs/BLUE-Lens"
REMOTE = "IntelLabs/BLUE-Lens"


def _download_features_from_hf(
    model: str,
    dataset: str,
    base_path: str = "blue_lens_hf_dataset",
    layer_id: Optional[int] = None,
    pattern: Optional[str] = None,
    subpattern: Optional[str] = None,
):
    repo_id = "IntelLabs/BLUE-Lens"
    """
    Downloads dataset files from Hugging Face Hub based on filtering parameters.

    Parameters:
    - repo_id (str): Hugging Face dataset repo ID.
    - model (str): Model name (e.g., 'gdino').
    - dataset (str): Dataset name (e.g., 'coco_mini').
    - base_path (str, optional): Base directory for downloading. Default: "blue_lens_hf_dataset".
    - layer_id (int, optional): Layer ID to filter files. Default: None.
    - pattern (str, optional): File pattern to match. Default: None.
    - subpattern (str, optional): Additional subpattern for finer filtering. Default: None.
    """

    # List all files in the dataset repository
    repo_files = list_repo_files(repo_id, repo_type="dataset")

    # Compute the correct dataset path
    dataset_path = f"features_datasets/{model}/{model}.{dataset}"

    # Filter files matching the dataset path
    filtered_repo_files = [file for file in repo_files if dataset_path in file]

    filter_path_parts = []
    if layer_id is not None:
        filter_path_parts.append(f"layer_{layer_id}")
    if pattern is not None:
        filter_path_parts.append(pattern)
    if subpattern is not None:
        filter_path_parts.append(subpattern)

    if filter_path_parts:
        files_to_download = filtered_repo_files
        for filter_part in filter_path_parts:
            files_to_download = [
                file for file in files_to_download if filter_part in file
            ]
    else:
        files_to_download = filtered_repo_files

    # Ensure at least some files are selected
    if not files_to_download:
        logger.error(f"⚠️ No matching files found in dataset")
        return

    # Ensure README.md and .gitattributes are included
    files_to_download.append(repo_files[0])  # '.gitattributes'
    files_to_download.append(repo_files[1])  # 'README.md'

    # Download the required files
    for file in files_to_download:
        local_file_path = os.path.join(base_path, file)  # Maintain structure
        # os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        hf_hub_download(
            repo_id=repo_id, filename=file, repo_type="dataset", local_dir=base_path
        )

    print(
        f"✅ Successfully downloaded {len(files_to_download) - 2} files in: {base_path}"
    )


def _build_feature_path_tree(
    dataset: Dataset, model: Model, sep: Optional[str] = None
) -> str:
    pieces = [
        "features_datasets",
        f"{model.lower()}",
        f"{model.lower()}.{dataset.lower()}",
    ]
    return sep.join(pieces) if sep else osp.join(*pieces)


def prepare_feature_disk_path(conf, name: str, dataset: Dataset, model: Model):
    ondisk_path = osp.join(
        conf.feature.path, _build_feature_path_tree(dataset, model), name
    )

    if osp.exists(ondisk_path):
        return ondisk_path

    remote_path = f"{_build_feature_path_tree(dataset, model, sep='/')}/{name}"
    if len(get_paths_info(REMOTE, remote_path, repo_type="dataset")) == 0:
        raise RuntimeError(f"No features found for provided remote path: {remote_path}")

    logger.info(
        f"Feature not found on disk, fetch\n"
        f"- from remote: {remote_path}\n"
        f"- to   ondisk: {ondisk_path}\n"
    )
    snapshot_download(
        REMOTE,
        allow_patterns=f"{remote_path}/*",
        repo_type="dataset",
        local_dir=conf.feature.path,
    )
    return ondisk_path


def fetch_remote_feature_names(
    conf: BLUEGLASSConf, dataset: Dataset, model: Model
) -> List[str]:
    filter_terms = _build_feature_path_tree(dataset, model)
    remote_names = list_repo_files(REMOTE, repo_type="dataset")
    filter_names = [name for name in remote_names if filter_terms in name]

    return list(set([name.split("/")[-2] for name in filter_names]))


def fetch_ondisk_feature_names(conf: BLUEGLASSConf, dataset: Dataset, model: Model):
    if conf.feature.path is None:
        return []

    search_path = osp.join(conf.feature.path, _build_feature_path_tree(dataset, model))

    if not osp.exists(search_path):
        logger.info("No features discovered on disk.")
        return []

    return os.listdir(search_path)
