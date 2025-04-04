# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List
import torch
from torch import Tensor
from itertools import compress


class Descriptions:
    def __init__(self, object_descriptions: List[str]):
        self.data = object_descriptions

    def __getitem__(self, item: Tensor):
        assert isinstance(item, Tensor), "expected a torch.Tensor"
        assert item.dim() == 1
        if len(item) > 0:
            assert item.dtype == torch.int64 or item.dtype == torch.bool
            if item.dtype == torch.int64:
                return Descriptions([self.data[int(x)] for x in item])
            elif item.dtype == torch.bool:
                return Descriptions(list(compress(self.data, item)))

        return Descriptions(list(compress(self.data, item)))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "Descriptions({})".format(self.data)
