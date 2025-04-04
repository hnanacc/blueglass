# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from fnmatch import fnmatchcase
from torch import nn
from blueglass.configs import BLUEGLASSConf, Model, FeaturePattern
from typing import List, Dict


def _florence_pattern_to_hooktmpl(name: str):
    match name:
        case FeaturePattern.LLM_DECODER_MHA:
            return "model.language_model.model.decoder.layers.*.self_attn"
        case FeaturePattern.LLM_DECODER_MLP:
            return "model.language_model.model.decoder.layers.*.fc2"
        case FeaturePattern.LLM_DECODER_RESID_MLP:
            return "model.language_model.model.decoder.layers.*.final_layer_norm"
        case FeaturePattern.LLM_DECODER_RESID_MHA:
            return "model.language_model.model.decoder.layers.*.?"
        case unsupported:
            raise NotImplementedError(f"Unsupported feature pattern: {unsupported}.")


def name_to_hookpattern(conf, name: str):
    match conf.model.name:
        case Model.FLORENCE:
            return _florence_pattern_to_hooktmpl(name)
        case unsupported:
            raise NotImplementedError(
                f"Unsupported model for pattern conversions: {unsupported}."
            )


def build_hooknames_from_names(
    conf: BLUEGLASSConf, model: nn.Module, used_names: List[str]
) -> Dict[str, List[str]]:
    hookpatterns = [name_to_hookpattern(conf, name) for name in used_names]
    compnames = [cn for cn, _ in model.named_modules()]
    return {
        name: [cn for cn in compnames if any([fnmatchcase(cn, hookpattern)])]
        for name, hookpattern in zip(used_names, hookpatterns)
    }
