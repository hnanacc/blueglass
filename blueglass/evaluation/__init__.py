# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from .label_matching import LabelMatchingEvaluator

from .multilayer import MultiLayerEvaluator
from .prediction_analysis import (
    PredictionAnalysisEvaluator as PredictionAnalysisEvaluator,
    compute_confusion_mask as compute_confusion_mask,
)
from .build import (
    build_dataset_specific_evaluator as build_dataset_specific_evaluator,
    build_mono_prediction_evaluator as build_mono_prediction_evaluator,
)
from .saes import SAEEvaluator
from blueglass.configs import BLUEGLASSConf


def build_evaluator(conf: BLUEGLASSConf, runner_mode: str = None) -> DatasetEvaluator:
    if conf.evaluator.use_multi_evaluators:
        evalautors = []
        for evaluator in conf.evaluator.names:
            evalautors.append(build_mono_prediction_evaluator(conf, runner_mode))
        return evalautors
    if conf.evaluator.use_multi_layer:
        return MultiLayerEvaluator(conf, runner_mode)
    else:
        return build_mono_prediction_evaluator(conf, runner_mode)
