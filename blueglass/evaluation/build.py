# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.third_party.detectron2.evaluation import (
    DatasetEvaluator,
    COCOEvaluator,
    LVISEvaluator,
)
from .bdd100k import BDD100kEvaluator
from .label_matching import LabelMatchingEvaluator
from .prediction_analysis import PredictionAnalysisEvaluator
from blueglass.configs import BLUEGLASSConf, Evaluator


def build_dataset_specific_evaluator(
    conf: BLUEGLASSConf, runner_mode: str = None
) -> DatasetEvaluator:

    conf_dataset = conf.dataset.infer if runner_mode == "infer" else conf.dataset.test
    match conf.evaluator.name:
        case Evaluator.COCO:
            return COCOEvaluator(
                conf_dataset, tasks=["bbox"], output_dir=conf.experiment.output_dir
            )
        case Evaluator.LVIS:
            return LVISEvaluator(
                conf_dataset, tasks=["bbox"], output_dir=conf.experiment.output_dir
            )
        case Evaluator.BDD100K:
            return BDD100kEvaluator(conf_dataset, conf.experiment.output_dir, 1)
        case unsupported:
            raise ValueError(f"unsupported evaluator type: {unsupported}")


def build_mono_prediction_evaluator(
    conf: BLUEGLASSConf, runner_mode: str = None
) -> DatasetEvaluator:
    de = build_dataset_specific_evaluator(conf)

    if conf.evaluator.use_label_matcher:
        de = LabelMatchingEvaluator(conf, de, runner_mode)

    if conf.evaluator.use_analysis:
        de = PredictionAnalysisEvaluator(conf, de, runner_mode)

    return de
