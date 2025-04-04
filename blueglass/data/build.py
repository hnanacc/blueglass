# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Union, Literal
from blueglass.third_party.detectron2.data.build import (
    get_detection_dataset_dicts,
    DatasetFromList,
    MapDataset,
    DatasetMapper,
    trivial_batch_collator,
    build_batch_data_loader,
)
from blueglass.third_party.detectron2.data.samplers import (
    InferenceSampler,
    TrainingSampler,
)

logger = setup_blueglass_logger(__name__)


def build_test_dataloader(
    dataset_name: str,
    batch_size: int = 1,
    num_workers: int = 4,
    evalaute_single_gpu=False,
    switch_dpp_on=False,
) -> DataLoader:
    logger.info("Building feature dataloader.")
    dataset = get_detection_dataset_dicts(
        dataset_name, filter_empty=False, proposal_files=None
    )
    assert isinstance(dataset, List), "unexpected data format for dataset."
    dataset = DatasetFromList(dataset)

    dataset = MapDataset(
        dataset, DatasetMapper(is_train=True, image_format="RGB", augmentations=[])
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=InferenceSampler(
            size=len(dataset), evalaute_single_gpu=evalaute_single_gpu
        ),
        shuffle=False,
        drop_last=False,
        collate_fn=trivial_batch_collator,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_train_dataloader(
    dataset_name, batch_size: int = 16, num_workers: int = 4
) -> DataLoader:
    logger.info("Building train dataloader.")
    dataset = get_detection_dataset_dicts(
        dataset_name, filter_empty=True, proposal_files=None
    )
    assert isinstance(dataset, List), "unexpected data format for dataset."
    dataset = DatasetFromList(dataset)
    dataset = MapDataset(
        dataset, DatasetMapper(is_train=True, image_format="RGB", augmentations=[])
    )
    return build_batch_data_loader(
        dataset,
        TrainingSampler(len(dataset)),
        batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
    )
