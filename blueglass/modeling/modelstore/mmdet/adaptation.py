# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os.path as osp
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import List, Dict, Any, Union
import torch
from torch import nn
from mmengine.registry import init_default_scope
from mmengine.dataset import pseudo_collate
from mmdet.structures import DetDataSample
from mmengine.config import Config as MMDetConfs
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.model.utils import revert_sync_batchnorm
from blueglass.third_party.detectron2.structures import Instances, Boxes, BoxMode
from mmdet.models.dense_heads import DINOHead, DETRHead
from mmengine.dataset import Compose
from mmdet.registry import MODELS
from blueglass.configs import BLUEGLASSConf, RunnerMode, Runner
from blueglass.third_party.detectron2.data import MetadataCatalog
from blueglass.third_party.detectron2.structures import Instances, Boxes

from blueglass.evaluation import compute_confusion_mask
from blueglass.configs import BLUEGLASSConf, FeaturePattern
from blueglass.features import intercept_manager

logger = setup_blueglass_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MMDET_SCOPE = "mmdet"


class MMDetModel(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__()

        self.conf = conf
        self.device = DEVICE
        init_default_scope(MMDET_SCOPE)

        self.mm_confs = self._prepare_mm_confs(conf)
        self.model = self._prepare_mm_model(conf, self.mm_confs)
        self.train_pipeline, self.test_pipeline = self._prepare_mm_pipelines(
            self.mm_confs
        )

    def _prepare_mm_confs(self, conf: BLUEGLASSConf) -> MMDetConfs:
        assert conf.model.conf_path is not None, "Require conf_path for mmdet models."
        assert osp.exists(conf.model.conf_path), "Conf file doens't exists."
        return MMDetConfs.fromfile(conf.model.conf_path)

    def _prepare_mm_model(self, conf: BLUEGLASSConf, mm_confs: MMDetConfs):
        model = MODELS.build(mm_confs.model)

        if conf.model.checkpoint_path:
            assert osp.exists(conf.model.checkpoint_path), "Checkpoint doesn't exists."
            logger.info("Found checkpoint for MMDet model.")
            model = self._with_checkpoint(conf.model.checkpoint_path, model)

            if (
                conf.runner.name is Runner.MODELSTORE
                and conf.runner.mode is RunnerMode.TRAIN
                and isinstance(model.bbox_head, DINOHead)
            ):
                logger.info("Reinitialized DINO class branches.")
                for mod in model.bbox_head.cls_branches:
                    ## initialising the bias similar to bias from COCO checkpoint
                    torch.nn.init.constant_(mod.bias, -4.59511985013459)
                    torch.nn.init.kaiming_uniform_(mod.weight)

        cnames = MetadataCatalog.get(conf.dataset.label).thing_classes
        setattr(model, "dataset_meta", {"classes": cnames})

        return model.to(DEVICE)

    def _with_checkpoint(self, path: str, model: nn.Module):
        revise_key = [(r"^module\.", ""), (r"^model\.", "")]
        checkpoint = _load_checkpoint(path, DEVICE)
        _load_checkpoint_to_model(model, checkpoint, revise_keys=revise_key)
        model = revert_sync_batchnorm(model)
        model.eval()
        return model

    def _prepare_mm_pipelines(
        self, mm_confs: MMDetConfs, ignore_transforms: List[str] = []
    ):
        trn_pipeline = [
            fn
            for fn in mm_confs.train_dataloader.dataset.pipeline
            if fn["type"] not in ignore_transforms
        ]
        tst_pipeline = [
            fn
            for fn in mm_confs.test_dataloader.dataset.pipeline
            if fn["type"] not in ignore_transforms
        ]
        return Compose(trn_pipeline), Compose(tst_pipeline)

    def preprocess(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pipeline = self.train_pipeline if self.training else self.test_pipeline
        return pseudo_collate([pipeline(self._d2_to_mm(bi)) for bi in batched_inputs])

    def forward(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        mm_pre = self.preprocess(batched_inputs)
        mm_pro = self.model.data_preprocessor(mm_pre, self.training)
        mm_pos = self.model(**mm_pro, mode=("loss" if self.training else "predict"))

        if self.training:
            _return = dict(self.model.parse_losses(mm_pos)[1])
        else:
            _return = self.postprocess(batched_inputs, mm_pos)

        return _return

    def _d2_to_mm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in inputs:
            del inputs["image"]

        if self.training:
            assert "instances" in inputs, "need true instances for train."

            boxes = inputs["instances"].gt_boxes.tensor.tolist()
            clses = inputs["instances"].gt_classes.tolist()

            insts = [
                {
                    "bbox": bx,
                    "bbox_label": cn,
                    "ignore_flag": False,
                }
                for bx, cn in zip(boxes, clses)
            ]
        else:
            insts = []

        return {
            "img_path": inputs["file_name"],
            "height": inputs["height"],
            "width": inputs["width"],
            "instances": insts,
        }

    def _mm_to_d2(self, inputs: Dict[str, Any], outputs: DetDataSample) -> Instances:
        mm_h, mm_w = outputs.get("ori_shape")
        assert (
            inputs["height"] == mm_h and inputs["width"] == mm_w
        ), "shape mismatch in input and output."
        inst = outputs.pred_instances
        return Instances(
            image_size=(mm_h, mm_w),
            pred_boxes=Boxes(inst.get("bboxes")),
            pred_classes=inst.get("labels"),
            scores=inst.get("scores"),
        )

    def postprocess(
        self, batched_inputs: List[Dict[str, Any]], batched_outputs: List[DetDataSample]
    ) -> List[Dict[str, Any]]:
        # return [
        #     {"instances": self._mm_to_d2(bi, bo)}
        #     for bi, bo in zip(batched_inputs, batched_outputs)
        # ]

        processed_batched_outputs = []
        for bi, bo in zip(batched_inputs, batched_outputs):

            new_inst = self._mm_to_d2(bi, bo)

            upd_boxes = Boxes(bo.pred_instances["bboxes"])
            assert isinstance(upd_boxes, Boxes), "unexpected conversion."

            processed_batched_outputs.append(
                {
                    "instances": new_inst,
                    "unprocessed_boxes": upd_boxes,
                    "unprocessed_bxind": bo.pred_instances["bboxes"],
                    "unprocessed_clsid": bo.pred_instances["labels"],
                    "unprocessed_score": bo.pred_instances["scores"],
                }
            )

        confusion_mask, pred_ious = compute_confusion_mask(
            self.conf, batched_inputs, processed_batched_outputs
        )
        intercept_manager().recorder(FeaturePattern.IO).record(
            "io",
            {
                "batched_inputs": batched_inputs,
                "batched_outputs": processed_batched_outputs,
                "confusion_mask": confusion_mask,
                "pred_ious": pred_ious,
            },
        )

        return processed_batched_outputs
