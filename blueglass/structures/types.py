# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

# write types for features, they are getting complex.
from enum import Enum
from torch import Tensor
from typing import List, Any, Dict, TypeGuard, TypeVar, Type, TypedDict

T = TypeVar("T")


def is_comm_dict(targets: List[Any]) -> TypeGuard[List[Dict[str, Any]]]:
    return all([isinstance(t, Dict) for t in targets])


def is_comm_list_of_dict(targets: List[Any]) -> TypeGuard[List[List[Dict[str, Any]]]]:
    return all([isinstance(t, List) for t in targets])


def is_list_of(type: Type[T], targets: Any) -> TypeGuard[List[T]]:
    return all([isinstance(t, type) for t in targets])


def is_comm_of(type: Type[T], targets: Any) -> TypeGuard[List[T]]:
    return all([isinstance(t, type) for t in targets])


def is_comm_number(targets: List[Any]) -> TypeGuard[List[float]]:
    return all([isinstance(t, float) or isinstance(t, int) for t in targets])
