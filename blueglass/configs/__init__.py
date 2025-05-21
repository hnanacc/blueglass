# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
from .defaults import BLUEGLASSConf as BLUEGLASSConf
from .utils import to_dict
from .constants import (
    Datasets,
    Model,
    Evaluator,
    FeaturePattern,
    FeatureSubPattern,
    Runner,
    RunnerMode,
    Precision,
    SAEVariant,
    ProbeVariant,
    Matcher,
    Prompter,
    Encoder,
    InterceptMode,
    DATASETS_AND_EVALS,
    WEIGHTS_DIR,
    MODELSTORE_CONFIGS_DIR,
    MODELSTORE_MMDET_CONFIGS_DIR,
    FEATURE_DIR,
)

from .defaults import (
    RunnerConf,
    DatasetConf,
    FeatureConf,
    ModelConf,
    EvaluatorConf,
    ProbeConf,
    ExperimentConf,
    LabelMatchEvaluatorConf,
    LayerKnockoffExpConf,
    SAEConf,
    SAEVariant,
)


from .modelstore_benchmarks import register_modelstore
from .modelstore_train import register_modelstore_train
from .features import register_features
from .probes import register_probes
from .saes import register_saes
from .interp import register_interp
from .knockoff_columns import register_layerknockoff
from .sae_knockoff import register_saeknockoff
from .sae_decoder_cluster import register_decoder_cluster


logger = setup_blueglass_logger(__name__)
# from .layers_patch import register_layerpatch


def register_all():
    def safe_register(register_function, name):
        try:
            register_function()
            logger.info(f"Successfully registered {name} Hydra based config.")
        except Exception as e:
            logger.warning(f"Failed to register {name} Hydra based config: {e}")

    safe_register(register_modelstore, "Modelstore")
    safe_register(register_modelstore_train, "Modelstore Train")
    safe_register(register_features, "Features")
    safe_register(register_probes, "Probes")
    safe_register(register_saes, "SAEs")
    safe_register(register_interp, "Interpretability")
    safe_register(register_layerknockoff, "Layer Knockoff")
    safe_register(register_saeknockoff, "SAE Knockoff")
    safe_register(register_decoder_cluster, "Decoder Clustering")


register_all()

__all__ = [
    "to_dict",
    "BLUEGLASSConf",
    "Datasets",
    "Model",
    "Evaluator",
    "FeaturePattern",
    "FeatureSubPattern",
    "Runner",
    "RunnerMode",
    "Precision",
    "SAEVariant",
    "ProbeVariant",
    "Matcher",
    "Prompter",
    "Encoder",
    "InterceptMode",
    "DATASETS_AND_EVALS",
    "WEIGHTS_DIR",
    "MODELSTORE_CONFIGS_DIR",
    "MODELSTORE_MMDET_CONFIGS_DIR",
    "FEATURE_DIR",
    "RunnerConf",
    "DatasetConf",
    "FeatureConf",
    "ModelConf",
    "EvaluatorConf",
    "ProbeConf",
    "ExperimentConf",
    "LabelMatchEvaluatorConf",
    "LayerKnockoffExpConf",
    "SAEConf",
    "SAEVariant",
]
