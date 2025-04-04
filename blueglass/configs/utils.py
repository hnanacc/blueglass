# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from .defaults import BLUEGLASSConf
from omegaconf import OmegaConf
from typing import Dict, Any, TypeGuard


def is_conf_dict(conf: Any) -> TypeGuard[Dict[str, Any]]:
    return isinstance(conf, Dict) and all(isinstance(key, str) for key in conf.keys())


def to_dict(conf: BLUEGLASSConf) -> Dict[str, Any]:
    transformed_conf = OmegaConf.to_container(conf, resolve=True, enum_to_str=True)
    assert is_conf_dict(transformed_conf), "Expected conf to be a dict."
    return transformed_conf
