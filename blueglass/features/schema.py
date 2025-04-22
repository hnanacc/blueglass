# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Union, Optional, Any
from dataclasses import dataclass
import pyarrow as pa
from torch import Tensor
from blueglass.configs import BLUEGLASSConf


@dataclass
class MinimalSchemaFrame:
    image_id: int
    infer_id: int
    heads_id: int
    token_id: int
    features: Tensor


@dataclass
class SchemaFrame(MinimalSchemaFrame):
    filename: str
    pred_box: Optional[List]
    pred_cls: Optional[int]
    pred_scr: Optional[float]
    pred_ious: Optional[float]
    conf_msk: Optional[bool]
    token_ch: Optional[str]


def build_arrow_schema(conf: BLUEGLASSConf, features_dim: int):
    return pa.schema(
        [
            ("image_id", pa.int32()),
            ("infer_id", pa.int32()),
            ("heads_id", pa.int32()),
            ("token_id", pa.int32()),
            ("filename", pa.string()),
            ("pred_box", pa.list_(pa.float32(), 4)),
            ("pred_cls", pa.int16()),
            ("pred_scr", pa.float32()),
            ("pred_ious", pa.float32()),
            ("conf_msk", pa.bool_()),
            ("token_ch", pa.string()),
            ("features", pa.list_(pa.float32(), features_dim)),
        ]
    )
