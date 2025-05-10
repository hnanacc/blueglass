# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict, Any
from collections import defaultdict
import torch
from torch import Tensor
from torchvision.ops import box_iou
from blueglass.third_party.detectron2.data import MetadataCatalog as MC
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from blueglass.configs import BLUEGLASSConf


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _process_instances(
    inputs: Dict[str, Any], outputs: Dict[str, Any]
) -> Dict[str, Tensor]:
    return {
        "gt_boxes": inputs["instances"].gt_boxes.tensor.to(DEVICE, dtype=torch.float),
        "dt_boxes": outputs["unprocessed_boxes"].tensor.to(DEVICE, dtype=torch.float),
        "gt_clsid": inputs["instances"].gt_classes.to(DEVICE),
        "dt_clsid": outputs["unprocessed_clsid"].to(DEVICE),
        "scores": outputs["unprocessed_score"].to(DEVICE, dtype=torch.float),
    }


def _compute_tp_and_fp(
    conf: BLUEGLASSConf, gt_boxes: Tensor, dt_boxes: Tensor, scores: Tensor
) -> Tensor:
    n_pred = dt_boxes.shape[0]
    cfmask = torch.zeros((n_pred,))

    if len(gt_boxes) == 0:
        return cfmask

    sorted_dt_boxid = torch.argsort(-scores)
    sorted_dt_boxes = dt_boxes[sorted_dt_boxid, :]

    gt_box_used = torch.zeros((len(gt_boxes),))
    dt_box_ious = box_iou(sorted_dt_boxes, gt_boxes)

    for p in range(n_pred):
        maxiou, maxind = dt_box_ious[p].max(dim=0)
        if maxiou > conf.evaluator.min_threshold_box:
            if not gt_box_used[maxind]:
                cfmask[sorted_dt_boxid[p]] = 1
                gt_box_used[maxind] = 1

    assert cfmask.sum() <= len(gt_boxes), "tp more than gt_boxes, unexpected."
    return cfmask


def compute_confusion_mask(
    conf: BLUEGLASSConf,
    batched_inputs: List[Dict[str, Any]],
    batched_outputs: List[Dict[str, Any]],
):
    """
    Computes True Positives (TP) and False Positives (FP)
    given the batched inputs and batched outputs.

    Returns
    --
    confusion_mask: List[Tensor], where 1 indicates TP and 0 indicates FP.
    """
    assert all(
        ["instances" in bi for bi in batched_inputs]
    ), "need instances in batched inputs."
    assert all(
        ["unprocessed_boxes" in bo for bo in batched_outputs]
    ), "need unprocessed boxes in batched outputs."
    assert all(
        ["unprocessed_clsid" in bo for bo in batched_outputs]
    ), "need unprocessed clsid in batched outputs."
    assert all(
        ["unprocessed_score" in bo for bo in batched_outputs]
    ), "need unprocessed scores in batched outputs."

    classnames = MC.get(conf.dataset.label).thing_classes
    batched_confusion_masks = []
    batched_pred_ious = []

    for bi, bo in zip(batched_inputs, batched_outputs):
        record = _process_instances(bi, bo)
        cfmask = torch.zeros((len(record["dt_boxes"]),))
        if len(record["gt_boxes"]) == 0:
            pred_ious = torch.zeros(len(record["dt_boxes"]))
            batched_confusion_masks.append(cfmask)
            batched_pred_ious.append(pred_ious)
            continue

        for cls_id in range(len(classnames)):
            gt_mask = record["gt_clsid"] == cls_id
            dt_mask = record["dt_clsid"] == cls_id

            cfmask_per_cls = _compute_tp_and_fp(
                conf,
                record["gt_boxes"][gt_mask],
                record["dt_boxes"][dt_mask],
                record["scores"][dt_mask],
            )
            cfmask[dt_mask] = cfmask_per_cls

        assert cfmask.sum() <= len(
            record["gt_boxes"]
        ), "unexpected, found more true positives than groundtruth."

        pred_ious = box_iou(record["dt_boxes"], record["gt_boxes"]).max(dim=-1).values
        batched_confusion_masks.append(cfmask)
        batched_pred_ious.append(pred_ious)

    return batched_confusion_masks, batched_pred_ious


class PredictionAnalysisEvaluator(DatasetEvaluator):
    def __init__(
        self,
        conf: BLUEGLASSConf,
        subevaluator: DatasetEvaluator,
        runner_mode: str = None,
    ):
        self.conf = conf
        self.subevaluator = subevaluator
        self.predic_intermed = defaultdict(list)

        self.compute_confusion = conf.evaluator.compute_confusion

    def reset(self):
        self.subevaluator.reset()
        self.predic_intermed = defaultdict(list)

    def process(self, batched_inputs, batched_outputs):
        self.subevaluator.process(batched_inputs, batched_outputs)

        if self.compute_confusion:
            self.predic_intermed["confusion_mask"].append(
                compute_confusion_mask(self.conf, batched_inputs, batched_outputs)
            )

    def evaluate(self):
        main_res = self.subevaluator.evaluate()

        # TODO: accumulate analysis.
        anls_res = self.predic_intermed

        assert isinstance(main_res, Dict), "invalid structure for results."
        assert isinstance(anls_res, Dict), "invalid structure for analysis."

        return {**main_res, **anls_res}
