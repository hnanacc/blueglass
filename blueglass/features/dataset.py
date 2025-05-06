# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import re
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
import torch.utils.data.datapipes.iter as pipes
from functools import lru_cache
from typing import Dict, Literal, Union, List, Any, Iterator
import pandas as pd
from collections import defaultdict, deque
from blueglass.configs import BLUEGLASSConf, Model, Datasets
from blueglass.utils.logger_utils import setup_blueglass_logger
from blueglass.data import build_test_dataloader
from blueglass.third_party.detectron2.utils import comm
from .interceptor import FeatureInterceptor
from .storage import FeatureStorage
from .types import DistFormat

logger = setup_blueglass_logger(__name__)


class FeatureStream:
    def __init__(
        self,
        conf: BLUEGLASSConf,
        model: Union[Model, nn.Module],
        dataset: Datasets,
        filter_scheme: str,
    ):
        self.conf = conf

        self.dataset = dataset
        self.model = model
        self.filter_scheme = filter_scheme
        self.source = None

    def mapper(self, batch_per_name: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("Override in child class.")

    def infer_feature_meta(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("Override in child class.")

    def fetch(self, filters: Dict[str, str]) -> Dict[str, Any]:
        raise NotImplementedError("Override in child class.")

    def __iter__(self) -> Iterator[Dict[str, Dict[str, Any]]]:
        raise NotImplementedError("Override in child class.")


class StorageStream(FeatureStream):
    def __init__(
        self,
        conf: BLUEGLASSConf,
        dataset: Datasets,
        model: Model,
        filter_scheme: str,
        batch_size: int,
    ):
        super().__init__(conf, model, dataset, filter_scheme)
        self.source = FeatureStorage(conf, dataset, model, filter_scheme, batch_size)

    def mapper(self, batch_per_name: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return self.source.format(batch_per_name)

    def infer_feature_meta(self) -> Dict[str, Any]:
        logger.debug(f"fetching feature meta data.")
        meta = self.source.infer_storage_meta()
        infered_feature_meta = {
            "feature_dim_per_name": {
                name: schema.field_by_name("features").type.list_size
                for name, schema in meta["feature_schemas"].items()
                if re.match(self.filter_scheme, name)
            },
            "num_records_per_name": meta["num_records_per_name"],
        }
        logger.info(f"fetched feature meta data.")
        return infered_feature_meta

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self.source


class InterceptorStream(FeatureStream):
    def __init__(
        self,
        conf,
        dataset: Datasets,
        model: nn.Module,
        filter_scheme: str,
        batch_size: int,
    ):
        super().__init__(conf, model, dataset, filter_scheme)
        assert isinstance(
            model, nn.Module
        ), "InterceptStream requires initialized model."

        self.dataset = dataset
        self.model = FeatureInterceptor(conf, model)
        self.batch_size = batch_size

        self.feature_meta = None
        self.buffer_per_name: Dict[str, deque] = defaultdict(deque)

    def mapper(self, batch_per_name: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return batch_per_name

    @lru_cache
    def infer_feature_meta(self) -> Dict[str, Any]:
        if self.feature_meta is None:
            dataloader = self._prepare_dataloader()
            batched_inputs = next(iter(dataloader))
            with torch.inference_mode():
                _, batched_features = self.model(batched_inputs, record=True)

            self.feature_meta = {
                "feature_dim_per_name": {
                    fname: len(frame.features[0])
                    for fname, frame in batched_features.items()
                    if re.match(self.filter_scheme, fname)
                },
                "num_records_per_name": {
                    fname: len(dataloader) * int(len(frame))
                    for fname, frame in batched_features.items()
                    if re.match(self.filter_scheme, fname)
                },
            }

        return self.feature_meta

    def __iter__(self) -> Iterator[Dict[str, Dict[str, Any]]]:
        for batched_inputs in self._prepare_dataloader():
            with torch.inference_mode():
                _, batched_features = self.model(batched_inputs, record=True)
            self._enqueue_buffer(self._filtered(batched_features))
            if self._is_buffer_reached():
                yield self._dequeue_buffer()

        while self._is_buffer_left():
            yield self._dequeue_buffer()

    def _prepare_dataloader(self) -> DataLoader:
        return build_test_dataloader(self.dataset, self.conf.dataset.batch_size)

    def _filtered(self, items: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: item
            for name, item in items.items()
            if re.match(self.filter_scheme, name)
        }

    def _is_buffer_reached(self) -> bool:
        return all(
            [len(buffer) >= self.batch_size for buffer in self.buffer_per_name.values()]
        )

    def _is_buffer_left(self) -> bool:
        return any([len(buffer) > 0 for buffer in self.buffer_per_name.values()])

    def _enqueue_buffer(self, batched_features: Dict[str, DistFormat]):
        for name, frames in batched_features.items():
            self.buffer_per_name[name].append(frames)

    def _batched(self, frame_chnks: List[DistFormat]) -> Dict[str, Any]:
        return pd.concat(frame_chnks, ignore_index=True).to_dict()  # type:ignore

    def _dequeue_per_name(self, buffer: deque) -> Dict[str, Any]:
        frame_chnks, frame_count = [], 0
        while len(buffer) > 0 and frame_count < self.batch_size:
            frame_chunk = buffer.popleft()
            frame_chnks.append(frame_chunk)
            frame_count += len(frame_chnks)
        return self._batched(frame_chnks)

    def _dequeue_buffer(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: self._dequeue_per_name(buffer)
            for name, buffer in self.buffer_per_name.items()
        }


class FeatureDataset(IterableDataset):
    """
    FeatureStore

    Unified API for storage and extraction-based feature fetch.

    Examples
    ---

    # Create the feature store for model and dataset.
    >>> fs = FeatureDataset(conf, model, dataset)

    >>> fs.infer_feature_meta()
    {
        layer_names:            ["layer.0", "layer.1", "layer.2"],
        feature_dim_per_layer:  [256, 256, 256],
        pattern_names:          ["det_decoder_resid_mlp", "det_decoder_mha"],
        subpattern_names:       [["pos_img", "refpnts"], ["weights", "outputs"]]
    }

    >>> fs.next_batch(
            filters={
                pattern_names=["det_decoder_resid_mlp"],
                layer_names=["layer.0"]
            }
        )
    {
        "batched_inputs": List[Dict],
        "batched_outputs": List[Dict],
        "batched_features": List[Tensor]
    }

    >>> fs.run(batched_inputs)
    batched_outputs: List[Dict]

    >>> fs.run_with_cache(batched_inputs, filters={...})
    (batched_outputs: List[Dict], batched_features: List[Dict])


    """

    def __init__(
        self,
        conf: BLUEGLASSConf,
        dataset: Datasets,
        model: Union[Model, nn.Module],
        local_mode: Literal["train", "test"] = "test",
        filter_scheme: str = r"layer_(\d+).(\w+).(\w+)",
    ):
        self.conf = conf
        self.dataset = dataset
        self.model = model
        self.local_mode = local_mode
        self.filter_scheme = filter_scheme
        if self.local_mode == "train":
            self.batch_size = conf.feature.train_batch_size
        else:
            self.batch_size = conf.feature.test_batch_size
        self.stream = (
            InterceptorStream(conf, dataset, model, filter_scheme, self.batch_size)
            if isinstance(model, nn.Module)
            else StorageStream(conf, dataset, model, filter_scheme, self.batch_size)
        )

    def __len__(self):
        if self.local_mode == "train":
            raise AttributeError("Infinite train loader doesn't provide length.")

        meta = self.stream.infer_feature_meta()
        assert "num_records_per_name" in meta and isinstance(
            meta["num_records_per_name"], Dict
        ), "Expected num records in feature meta."

        num_records = set(meta["num_records_per_name"].values())
        assert len(num_records) == 1, "Num records vary per name."

        return math.ceil(num_records.pop() / self.batch_size)

    def mapper(self, batch_per_name: Dict[str, Any]):
        return self.stream.mapper(batch_per_name)

    def infer_feature_meta(self) -> Dict[str, Any]:
        """
        Returns a dict of layer names and feature dimensions.
        """
        return self.stream.infer_feature_meta()

    def _infer_batch_size(self, batch_per_name: Dict[str, Any]):
        batch_sizes = set([len(batch) for batch in batch_per_name.values()])
        assert len(batch_sizes) == 1, "batch sizes differ for each name."
        return batch_sizes.pop()

    def __iter__(self):
        match self.local_mode:
            case "train":
                while True:
                    for batch_per_name in self.stream:
                        if self._infer_batch_size(batch_per_name) < self.batch_size:
                            continue
                        yield batch_per_name
            case "test":
                yield from self.stream
            case unsupported:
                raise ValueError(f"Unsupported mode: {unsupported}.")


def build_feature_dataloader(
    conf: BLUEGLASSConf,
    dataset: Datasets,
    model: Union[Model, nn.Module],
    local_mode: Literal["train", "test"],
    filter_scheme: str = r"layer_(\d+).(\w+).(\w+)",
    num_workers: int = 1,
) -> DataLoader:
    ds = FeatureDataset(conf, dataset, model, local_mode, filter_scheme)
    mp = ds.mapper

    if isinstance(model, Model):
        ds = (
            pipes.IterableWrapper(ds)
            .shuffle(buffer_size=64)
            .sharding_filter()
            .map(lambda batch: mp(batch))
        )

    if local_mode == "train":
        return DataLoader(
            ds,
            shuffle=True,
            batch_size=None,
            num_workers=1,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=1,
        )
    else:
        return DataLoader(ds, shuffle=False, batch_size=None)
