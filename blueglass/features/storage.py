# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import re
import os
import os.path as osp
import math
import shutil
from blueglass.utils.logger_utils import setup_blueglass_logger
import pandas as pd
from collections import deque
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import torch
from collections import OrderedDict
from blueglass.configs import BLUEGLASSConf, Model, Datasets
from blueglass.utils.feature import (
    fetch_ondisk_feature_names,
    fetch_remote_feature_names,
    prepare_feature_disk_path,
)
from typing import Dict, Any, Optional, List, Iterator
from .schema import build_arrow_schema
from .types import DistFormat

logger = setup_blueglass_logger(__name__)


class Reader:
    def __init__(self, conf: BLUEGLASSConf, name: str, dataset: Datasets, model: Model):
        self.batch_size = conf.feature.batch_size
        self.path = prepare_feature_disk_path(conf, name, dataset, model)
        self.stream = ds.dataset(self.path, format="parquet")

    def infer_schema(self) -> pa.Schema:
        return self.stream.schema

    def num_records(self) -> int:
        return self.stream.count_rows(batch_size=1)

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        yield from self.stream.to_batches(batch_size=self.batch_size)


class Writer:
    def __init__(self, conf: BLUEGLASSConf, name: str, dataset: Datasets, model: Model):
        self.conf = conf
        self.max_row_count = conf.feature.max_rows_per_part
        self.cur_row_count = 0
        self.num_committed = 0
        self._buffer: deque[DistFormat] = deque()

        assert conf.feature.path is not None, "Require path to write."

        self.base_dir_path = osp.join(
            conf.feature.path,
            "features_datasets",
            f"{model.lower()}",
            f"{model.lower()}.{dataset.lower()}",
            name,
        )

        if os.path.exists(self.base_dir_path):
            logger.info("Base writer path already exists. Overwritten.")
            shutil.rmtree(self.base_dir_path)

        os.makedirs(self.base_dir_path)

        logger.info(f"Saving features: {name} in {self.base_dir_path}.")

        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None

    def _create_writer(self, part: int) -> pq.ParquetWriter:
        assert self._schema is not None, "Schema must be initialized before."
        return pq.ParquetWriter(
            osp.join(self.base_dir_path, f"part.{part}"),
            self._schema,
            compression="zstd",
            compression_level=7,
        )

    def _prepare_schema(self, records: DistFormat):
        if self._schema:
            return
        features_dim = len(records.features[0])
        self._schema = build_arrow_schema(self.conf, features_dim)

    def _buffer_records(self, records: DistFormat):
        self._buffer.append(records)
        self.cur_row_count += len(records)

    def _buffer_reached(self):
        return self.cur_row_count >= self.max_row_count

    def _write_part(self, records: List[DistFormat]):
        if sum([len(r) for r in records]) == 0:
            return

        wframe = pd.concat(records, ignore_index=True)
        writer = self._create_writer(self.num_committed)
        writer.write(pa.Table.from_pandas(wframe, schema=self._schema))
        writer.close()

        self.cur_row_count -= len(wframe)
        self.num_committed += 1

    def _commit_buffer(self):
        while self.cur_row_count >= self.max_row_count:
            par_row_count = 0
            par_row_frame = []

            while par_row_count < self.max_row_count:
                frame = self._buffer.popleft()
                rsize = self.max_row_count - par_row_count
                ssize = min(len(frame), rsize)
                prows = frame[:ssize]
                frame = frame[ssize:]

                par_row_frame.append(prows)
                par_row_count += len(prows)

                if len(frame) > 0:
                    self._buffer.appendleft(frame)

            assert par_row_count == self.max_row_count, "Unexpected: size mismatch."
            self._write_part(par_row_frame)

    def stream_write(self, records: DistFormat):
        self._prepare_schema(records)
        self._buffer_records(records)

        if self._buffer_reached():
            self._commit_buffer()

    def __del__(self):
        assert (
            self.cur_row_count <= self.max_row_count
        ), "Unexpected: buffer greater than expected while flush."
        self._write_part(list(self._buffer))
        logger.info(f"Flushed out buffer of size: {self.cur_row_count}")


class FeatureStorage:
    def __init__(
        self,
        conf: BLUEGLASSConf,
        dataset: Datasets,
        model: Model,
        filter_scheme: str = r"layer_(\d+).(\w+).(\w+)",
    ):
        self.conf = conf
        self.dataset = dataset
        self.model = model
        self.filter_scheme = filter_scheme
        self.storage_meta: Optional[Dict[str, Any]] = None
        self.reader_per_name: OrderedDict[str, Reader] = OrderedDict()
        self.writer_per_name: OrderedDict[str, Writer] = OrderedDict()

    def _convert_to_torch(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        tensor_attrs = [
            "features",
            "pred_box",
            "pred_cls",
            "pred_scr",
            "pred_ious",
            "conf_msk",
        ]
        for attr in tensor_attrs:
            batch[attr] = torch.as_tensor(batch[attr])
        return batch

    def format(self, batch_per_name: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        for name, batch in batch_per_name.items():
            if isinstance(batch, pa.RecordBatch):
                batch_per_name[name] = self._convert_to_torch(
                    batch.to_pandas().to_dict("list")
                )

        return batch_per_name

    def infer_storage_meta(self) -> Dict[str, Dict[str, Any]]:
        if self.storage_meta is None:
            self.storage_meta = self._prepare_storage_meta()
        return self.storage_meta

    def write(self, records_per_name: Dict[str, pd.DataFrame]):
        for name, records in records_per_name.items():
            if name not in self.writer_per_name:
                self.writer_per_name[name] = Writer(
                    self.conf, name, self.dataset, self.model
                )
            self.writer_per_name[name].stream_write(records)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._prepare_readers()
        for batch_per_name in zip(*list(self.reader_per_name.values())):
            yield {
                name: batch
                for name, batch in zip(self.reader_per_name.keys(), batch_per_name)
            }

    def _discover_feature_names(self) -> List[str]:
        discovered = set(
            [
                *fetch_remote_feature_names(self.conf, self.dataset, self.model),
                *fetch_ondisk_feature_names(self.conf, self.dataset, self.model),
            ]
        )
        discovered = [
            name for name in sorted(discovered) if re.match(self.filter_scheme, name)
        ]

        if len(discovered) == 0:
            raise RuntimeError("No features found for provided configuration.")

        return discovered

    def _prepare_readers(self):
        discover_feature_names = self._discover_feature_names()
        for name in discover_feature_names:
            if name not in self.reader_per_name:
                _reader_per_name = Reader(self.conf, name, self.dataset, self.model)
                check_rows = _reader_per_name.stream.count_rows(batch_size=1)
                if check_rows == 0:
                    logger.warning(
                        f"Pattern '{name}' has no data - skipping reader registration."
                    )
                else:
                    self.reader_per_name[name] = _reader_per_name

    def _prepare_storage_meta(self) -> Dict[str, Dict[str, Any]]:
        self._prepare_readers()
        storage_meta = {
            "feature_schemas": {
                name: reader.infer_schema()
                for name, reader in self.reader_per_name.items()
            },
            "num_records_per_name": {
                name: reader.num_records()
                for name, reader in self.reader_per_name.items()
            },
        }
        logger.info(f"Completed building all readers.")
        return storage_meta
