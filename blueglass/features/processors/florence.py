# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from .base_processor import Processor

from typing import Callable, List, Dict, Any, Union, TypedDict
from collections import defaultdict
import torch
from torch import Tensor

from ..schema import MinimalSchemaFrame
from ..accessors import Recorder


class FlorenceProcessor(Processor):
    pass
