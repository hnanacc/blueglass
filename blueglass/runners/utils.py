# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from torch import nn
from typing import Iterable, List, Dict, Any, Tuple, Union
class BestTracker:
    def __init__(self):
        self.best_fitness = None
        self.best_step = None

    def is_best(self, fitness: float, step: int) -> bool:
        is_best = self.best_fitness is None or fitness > self.best_fitness

        if is_best:
            self.best_fitness = fitness
            self.best_step = step

        return is_best

    def best(self) -> float:
        if self.best_fitness is not None:
            return self.best_fitness
        else:
            return 0.0


def maybe_strip_ddp(
    model: Union[nn.Module, nn.parallel.DistributedDataParallel],
    ) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model