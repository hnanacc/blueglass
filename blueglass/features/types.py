# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from torch import Tensor


@dataclass
class MHAFrame:
    weights: Optional[List[Tensor]] = None
    outputs: Optional[List[Tensor]] = None


@dataclass
class MLPFrame:
    pre_img: Optional[List[Tensor]] = None
    pos_img: Optional[List[Tensor]] = None
    refpnts: Optional[List[Tensor]] = None


@dataclass
class ResidFrame:
    pre_img: Optional[List[Tensor]] = None
    pos_img: Optional[List[Tensor]] = None
    pre_txt: Optional[List[Tensor]] = None
    pos_txt: Optional[List[Tensor]] = None
    refpnts: Optional[List[Tensor]] = None


@dataclass
class IOFrame:
    image_id: List[str]
    filename: List[str]
    pred_box: Optional[List[list]] = None
    pred_cls: Optional[List[list]] = None
    pred_scr: Optional[List[list]] = None
    conf_msk: Optional[List[list]] = None
    token_ch: Optional[List[int]] = None
    pred_ious: Optional[List[list]] = None


LayerFrame = Union[MHAFrame, MLPFrame, ResidFrame, IOFrame]

DistFormat = pd.DataFrame

FilterParams = Dict[str, Any]
