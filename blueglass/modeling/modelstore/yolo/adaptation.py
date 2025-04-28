# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict, Any
import torch
from torch import nn, inference_mode
from ultralytics import YOLO
from ultralytics.engine.results import Results
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from blueglass.configs import BLUEGLASSConf


class YOLO_X(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        from ultralytics import YOLO

        super().__init__()
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert conf.model.checkpoint_path is not None, "missing checkpoint path."
        self.model = YOLO(conf.model.checkpoint_path, task="detect").to(self.device)
        assert self.model.model is not None

    def eval(self):
        assert isinstance(self.model.model, nn.Module), "submodel not initialized."
        return self.model.model.eval()

    def train(self, mode):
        assert isinstance(self.model.model, nn.Module), "submodel not initialized."
        return self.model.model.train(mode)

    @inference_mode()
    def forward(self, batched_inputs: List[Dict[str, Any]]):
        images = self.preprocess(batched_inputs)
        batched_outputs = self.model.predict(
            images,
            conf=self.conf.evaluator.min_threshold_cls,
            max_det=self.conf.evaluator.max_predictions,
        )
        return self.postprocess(batched_inputs, batched_outputs)

    def preprocess(self, batched_inputs: List[Dict[str, Any]]) -> List[str]:
        return [bi["file_name"] for bi in batched_inputs]

    def postprocess(
        self, batched_inputs: List[Dict[str, Any]], batched_outputs: List[Results]
    ):
        processed_batched_outputs = []
        for _, bo in zip(batched_inputs, batched_outputs):
            assert bo.boxes is not None, "boxes not enabled."
            inst = Instances(bo.boxes.orig_shape)

            assert isinstance(bo.boxes.xyxy, torch.Tensor), "unexpected format."
            inst.pred_boxes = Boxes(bo.boxes.xyxy)

            inst.pred_box_ious = torch.ones(
                len(bo.boxes), dtype=torch.float32, device=self.device
            )
            inst.pred_box_objectness = bo.boxes.conf
            inst.pred_classes = bo.boxes.cls
            inst.pred_object_descriptions = Descriptions(
                [bo.names[int(cid)] for cid in inst.pred_classes]
            )
            processed_batched_outputs.append({"instances": inst})
        return processed_batched_outputs
