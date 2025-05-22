# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import json
from enum import Enum
from dataclasses import fields, is_dataclass
import json
from enum import Enum
from dataclasses import fields, is_dataclass
from omegaconf import OmegaConf
from typing import Dict, Any, TypeGuard
from typing import Any, Type, List
from .defaults import BLUEGLASSConf
from .defaults import (
    SAEVariant,
    FeaturePattern,
    FeatureSubPattern,
    Datasets,
    Model,
    Evaluator,
)


def is_conf_dict(conf: Any) -> TypeGuard[Dict[str, Any]]:
    return isinstance(conf, Dict) and all(isinstance(key, str) for key in conf.keys())


def to_dict(conf: BLUEGLASSConf) -> Dict[str, Any]:
    transformed_conf = OmegaConf.to_container(conf, resolve=True, enum_to_str=True)
    assert is_conf_dict(transformed_conf), "Expected conf to be a dict."
    return transformed_conf


def load_blueglass_from_wandb(
    json_path: str, original_config: BLUEGLASSConf = None
) -> Any:
    """
    Load a WandB-style config JSON into a structured config dataclass instance.
    Automatically unwraps {"value": ...} entries and skips nulls.
    Returns a fully-typed BLUEGLASSConf dataclass.

    Usage: cfg = load_blueglass_from_wandb("wandb_config.json"
    """

    def unwrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            if set(obj.keys()) == {"value"}:
                return unwrap(obj["value"])
            return {k: unwrap(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [unwrap(i) for i in obj]
        elif obj is None:
            return None
        else:
            return obj

    def recursive_merge(instance: Any, overrides: dict) -> Any:
        for f in fields(instance):
            key = f.name
            if key in overrides:
                current_val = getattr(instance, key)
                override_val = overrides[key]
                if is_dataclass(current_val) and isinstance(override_val, dict):
                    setattr(instance, key, recursive_merge(current_val, override_val))
                else:
                    setattr(instance, key, override_val)
        return instance

    with open(json_path, "r") as f:
        raw_json = json.load(f)

    clean_json = unwrap(raw_json)
    default_instance = BLUEGLASSConf()

    blueglassconf = recursive_merge(default_instance, clean_json)

    if isinstance(blueglassconf.sae.variant, str):
        blueglassconf.sae.variant = SAEVariant(blueglassconf.sae.variant.lower())

    pattern = blueglassconf.feature.patterns
    if isinstance(pattern, list):
        blueglassconf.feature.patterns = [
            FeaturePattern(p.lower()) if isinstance(p, str) else p for p in pattern
        ]
    else:
        blueglassconf.feature.patterns = (
            FeaturePattern(pattern.lower()) if isinstance(pattern, str) else pattern
        )

    sub_pattern = blueglassconf.feature.sub_patterns
    if isinstance(sub_pattern, list):
        blueglassconf.feature.sub_patterns = [
            FeatureSubPattern(p.lower()) if isinstance(p, str) else p
            for p in sub_pattern
        ]
    else:
        blueglassconf.feature.sub_patterns = (
            FeatureSubPattern(sub_pattern.lower())
            if isinstance(sub_pattern, str)
            else sub_pattern
        )

    if isinstance(blueglassconf.dataset.infer, str):
        blueglassconf.dataset.infer = Datasets(blueglassconf.dataset.infer.lower())
    if isinstance(blueglassconf.dataset.label, str):
        blueglassconf.dataset.label = Datasets(blueglassconf.dataset.label.lower())
    if isinstance(blueglassconf.dataset.test, str):
        blueglassconf.dataset.test = Datasets(blueglassconf.dataset.test.lower())
    if isinstance(blueglassconf.dataset.train, str):
        blueglassconf.dataset.train = Datasets(blueglassconf.dataset.train.lower())
    if isinstance(blueglassconf.model.name, str):
        blueglassconf.model.name = Model(blueglassconf.model.name.lower())

    if isinstance(blueglassconf.evaluator.names, str):
        blueglassconf.evaluator.name = Evaluator(blueglassconf.evaluator.name.lower())

    if isinstance(blueglassconf.evaluator.names, List):
        names = blueglassconf.evaluator.names
        blueglassconf.evaluator.names = [
            Evaluator(name) if isinstance(name, str) else name for name in names
        ]

    if original_config is not None:
        # Merge with the original config
        blueglassconf.feature.path = original_config.feature.path
        blueglassconf.model.conf_path = original_config.model.conf_path
        blueglassconf.model.checkpoint_path = original_config.model.checkpoint_path
    return blueglassconf
