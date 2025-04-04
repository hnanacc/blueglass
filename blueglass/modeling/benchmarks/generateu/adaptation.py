# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.third_party.detectron2.data import transforms as T
from blueglass.evaluation import compute_confusion_mask
from blueglass.configs import BLUEGLASSConf, FeaturePattern
from blueglass.features import intercept_manager
from blueglass.evaluation import (
    build_dataset_specific_evaluator,
    LabelMatchingEvaluator,
)


class GenerateU(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        super().__init__()

        self.conf = conf
        self.label_matching_evaluator = LabelMatchingEvaluator(
            self.conf, build_dataset_specific_evaluator(self.conf)
        )
        assert (
            conf.model.checkpoint_path is not None
        ), "missing checkpoint_path for GenerateU."

        cfg, output_dir = self.setup(conf)
        self.model = build_model(cfg).eval()
        DetectionCheckpointer(self.model, save_dir=output_dir).resume_or_load(
            conf.model.checkpoint_path, resume=False
        )
        self.transforms = T.ResizeShortestEdge([800], 1333, "choice")
        self.device = torch.device("cuda")

        assert cfg.UNI, "model cannot be used in un-unified mode."

    def setup(self, conf: BLUEGLASSConf):
        from detectron2.config import get_cfg
        from detectron2.projects.ddetrs import add_ddetrsvluni_config

        assert conf.model.conf_path is not None, "missing conf_path for GenerteU."
        assert (
            conf.model.checkpoint_path is not None
        ), "missing checkpoint_path for GenerteU."
        assert (
            conf.model.checkpoint_path_genu_embed is not None
        ), "missing checkpoint_path_genu_embed for GenerteU."

        cfg = get_cfg()
        add_ddetrsvluni_config(cfg)
        cfg.merge_from_file(conf.model.conf_path)
        cfg.UNI = True
        cfg.MODEL.WEIGHTS = conf.model.checkpoint_path
        cfg.MODEL.TEXT.ZERO_SHOT_WEIGHT = conf.model.checkpoint_path_genu_embed
        cfg.TEST.EVALUATOR_TYPE = "custom"
        cfg.freeze()
        return cfg, conf.experiment.output_dir

    @torch.inference_mode()
    def forward(self, batched_inputs: List[Dict]):
        batched_inputs = self.preprocess(batched_inputs)
        batched_outputs = self.model(batched_inputs)
        return self.postprocess(batched_inputs, batched_outputs)

    def transform(self, im: torch.Tensor):
        np_im = np.asarray(im).transpose(1, 2, 0)
        np_im = self.transforms.get_transform(np_im).apply_image(np_im)
        return torch.as_tensor(np.ascontiguousarray(np_im.transpose(2, 0, 1)))

    def preprocess(self, batched_inputs: List[Dict]):
        for bi in batched_inputs:
            bi["task"] = bi["test_prompt"] = "detect"
            bi["image"] = self.transform(bi["image"])
            bi.pop("annotations", None)
        return batched_inputs

    def postprocess(self, batched_inputs: List[Dict], batched_outputs: List[Dict]):

        processed_batched_outputs = []
        for bi, bo in zip(batched_inputs, batched_outputs):
            new_inst = Instances(
                bo["instances"].image_size, **bo["instances"].get_fields()
            )
            new_inst.pred_boxes = Boxes(new_inst.pred_boxes.tensor)
            bo["instances"] = new_inst
            """
            Since Conf mask is needed to be computed at this stage to maintain sanity, we initialise the label matching evaluator only to convert descriptions into class ids with respect to the dataset used in the conf.
            sub_eval is set to false because we just need the processed output for confusion mask and sub_eval is used only for complete evaluation with metrics as outputs.
            """
            ebo = self.label_matching_evaluator.process([bi], [bo], sub_eval=False)[0]
            processed_batched_outputs.append(
                {
                    "instances": new_inst,
                    "unprocessed_boxes": ebo["instances"].pred_boxes,
                    "unprocessed_bxind": None,
                    "unprocessed_clsid": ebo["instances"].pred_classes,
                    "unprocessed_score": ebo["instances"].scores,
                }
            )

        intercept_manager().recorder(FeaturePattern.IO).record(
            "io",
            {
                "batched_inputs": batched_inputs,
                "batched_outputs": processed_batched_outputs,
                "confusion_mask": compute_confusion_mask(
                    self.conf, batched_inputs, processed_batched_outputs
                ),
            },
        )
        return processed_batched_outputs
