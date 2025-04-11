# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
from .defaults import (
    BLUEGLASSConf as BLUEGLASSConf,
    Dataset as Dataset,
    Model as Model,
    Evaluator as Evaluator,
    FeaturePattern as FeaturePattern,
    FeatureSubPattern as FeatureSubPattern,
    Runner as Runner,
    RunnerMode as RunnerMode,
    Precision as Precision,
    SAEVariant as SAEVariant,
    ProbeVariant as ProbeVariant,
    Matcher as Matcher,
    Prompter as Prompter,
    Encoder as Encoder,
    InterceptMode as InterceptMode,
)
from .utils import to_dict
from .modelstore import register_modelstores
from .features import register_features
from .probes import register_probes
from .saes import register_saes
from .interp import register_interp
from .knockoff_red_attn_wt import register_layerknockoff


logger = setup_blueglass_logger(__name__)
# from .layers_patch import register_layerpatch


def register_all():
    def safe_register(register_function, name):
        try:
            register_function()
            logger.info(f"Successfully registered {name} Hydra based config.")
        except Exception as e:
            logger.warning(f"Failed to register {name} Hydra based config: {e}")

    safe_register(register_modelstores, "Modelstore")
    safe_register(register_features, "Features")
    safe_register(register_probes, "Probes")
    safe_register(register_saes, "SAEs")
    safe_register(register_interp, "Interpretability")
    safe_register(register_layerknockoff, "Layer Knockoff")


register_all()
