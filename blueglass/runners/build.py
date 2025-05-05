# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
from blueglass.configs import BLUEGLASSConf, Runner as RunnerType
from .runner import Runner
from .modelstore import ModelstoreRunner
from .layers_patch import LayersPatchRunner
from .features_extract import FeatureExtractRunner
from .probes_linear_vlm import VLMLinearProbeRunner
from .sae_probe import SAELinearProbeRunner
from .sae_interp import InterpretationRunner
from .saes import SAERunner

logger = setup_blueglass_logger(__name__)


def build_runner(conf: BLUEGLASSConf) -> Runner:
    runner_classes = {
        RunnerType.MODELSTORE: ModelstoreRunner,
        RunnerType.FEATURE_EXTRACT: FeatureExtractRunner,
        RunnerType.VLM_LINEAR_PROBE: VLMLinearProbeRunner,
        RunnerType.SAE_LINEAR_PROBE: SAELinearProbeRunner,
        RunnerType.SAE: SAERunner,
        RunnerType.LAYERS_PATCH: LayersPatchRunner,
        RunnerType.INTERPRETATION: InterpretationRunner,
    }

    runner_class = runner_classes.get(conf.runner.name)
    if runner_class is None:
        raise NotImplementedError(f"Unsupported runner: {conf.runner.name}.")

    logger.debug(f"Building the runner {conf.runner.name}.")
    runner = runner_class(conf)
    logger.info(f"Successfully registered runner {conf.runner.name}.")
    return runner
