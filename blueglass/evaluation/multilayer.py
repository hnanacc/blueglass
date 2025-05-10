# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from copy import deepcopy
from typing import Dict, List, Any
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from blueglass.evaluation.build import build_mono_prediction_evaluator
from blueglass.configs import BLUEGLASSConf


class MultiLayerEvaluator(DatasetEvaluator):
    def __init__(self, conf: BLUEGLASSConf, runner_mode: str = None):
        self.conf = conf
        self.output_dir = f"{conf.experiment.output_dir}/inference"
        self.subevaluators: Dict[str, DatasetEvaluator] = {}

    def build_layer_evaluator(self, layer_name: str):
        conf = deepcopy(self.conf)
        conf.experiment.output_dir = f"{self.output_dir}/{layer_name}"
        return build_mono_prediction_evaluator(conf)

    def reset(self):
        for evaluator in self.subevaluators.values():
            evaluator.reset()

    def process(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_outputs: Dict[str, List[Dict[str, Any]]],
    ):
        for layer_name, batched_outputs_per_layer in batched_outputs.items():
            if layer_name not in self.subevaluators:
                self.subevaluators[layer_name] = self.build_layer_evaluator(layer_name)
                self.subevaluators[layer_name].reset()

            self.subevaluators[layer_name].process(
                batched_inputs, batched_outputs_per_layer
            )

    def evaluate(self):
        return {
            layer_name: evaluator.evaluate()
            for layer_name, evaluator in self.subevaluators.items()
        }
