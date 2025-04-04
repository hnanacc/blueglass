# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Dict, List, Tuple, Any, Literal
from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import F as L
from torchvision.ops import (
    box_convert,
    generalized_box_iou_loss,
    sigmoid_focal_loss,
)
from blueglass.third_party.detectron2.data import MetadataCatalog as MC


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_boxes(
    batched_boxes: List[Tensor],
    im_sizes: List[Tuple[int, int]],
    to: Literal["rel", "abs"],
) -> List[Tensor]:
    assert to in ["abs", "rel"], "invalid mode for normalization."

    normalized_boxes = []

    for boxes, (im_h, im_w) in zip(batched_boxes, im_sizes):
        factor = (
            boxes.new_tensor([im_w, im_h, im_w, im_h])
            .unsqueeze(0)
            .repeat(boxes.size(0), 1)
        )

        if to == "abs":
            boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy") * factor
        if to == "rel":
            boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh") / factor

        normalized_boxes.append(boxes)

    return normalized_boxes


class ApproximateVLMLoss:
    """
    Computes (1:1) losses between the predictions of linear probe
    and the predictions of VLM.

    Goal is to see if an intermediate layer can approximate, the
    outputs of the final layers including the false positives.

    Why? To see what layers are absolutely useful.
    """

    def __init__(self, args):
        self.device = DEVICE
        self.probe_mode = args.probe_mode
        self.num_spatial_units = int(args.num_spatial_units)
        self.num_classes = len(MC.get(args.labelset).thing_classes)
        self.use_classified_boxes = bool(args.use_classified_boxes)

    def __call__(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_feats_outputs,
        batched_probe_outputs: Dict[str, List[Tensor]],
    ) -> Dict[str, Any]:

        losses, io = {}, batched_feats_outputs.fetch_io()

        if "pred_box" in batched_probe_outputs:
            im_sizes = [(bi["height"], bi["width"]) for bi in batched_inputs]

            # Step 1: Format dt/gt boxes and classes.
            batched_prb_box = [pb for pb in batched_probe_outputs["pred_box"]]
            batched_vlm_box = normalize_boxes(
                [bf for bf in io["PRED_BOX"]],
                im_sizes,
                to="rel",
            )

            if self.use_classified_boxes:
                batched_prb_box = [pb.softmax(dim=-1) for pb in batched_prb_box]
                batched_vlm_box = [
                    F.one_hot(
                        (box * self.num_spatial_units).long(),
                        self.num_spatial_units,
                    ).float()
                    for box in batched_vlm_box
                ]
            else:
                batched_prb_box = [pb.sigmoid() for pb in batched_prb_box]

            # Step 2: Flatten preds and targs.
            prb_box = torch.cat(batched_prb_box, dim=0)
            vlm_box = torch.cat(batched_vlm_box, dim=0)

            # Step 3: Compute losses for boxes.
            if self.use_classified_boxes:
                losses["loss_box"] = L.binary_cross_entropy_with_logits(
                    prb_box, vlm_box
                )
            else:
                losses["loss_box"] = L.smooth_l1_loss(prb_box, vlm_box)

        if "pred_cls" in batched_probe_outputs:

            # Step 1: Format dt/gt boxes and classes.
            batched_prb_cls = [pc for pc in batched_probe_outputs["pred_cls"]]
            batched_vlm_cls = [
                F.one_hot(pc.long(), self.num_classes).float()
                for pc in io["IOPatterns"]
            ]

            # Step 2: Flatten preds and targs.
            prb_cls = torch.cat(batched_prb_cls, dim=0)
            vlm_cls = torch.cat(batched_vlm_cls, dim=0)

            # Step 3: Compute losses for class.
            losses["loss_cls"] = L.binary_cross_entropy_with_logits(prb_cls, vlm_cls)

        if "pred_iou" in batched_probe_outputs:
            im_sizes = [(bi["height"], bi["width"]) for bi in batched_inputs]

            # Step 1: Extract confusion mask.
            batched_prb_iou = [
                bi.squeeze().sigmoid() for bi in batched_probe_outputs["pred_iou"]
            ]
            batched_vlm_iou = [bf.to(self.device) for bf in io["CONF_MSK"]]

            # Step 2: Flatten preds and targs.
            prb_iou = torch.cat(batched_prb_iou, dim=0)
            vlm_iou = torch.cat(batched_vlm_iou, dim=0)

            # Step 3: Compute losses for boxes.
            losses["loss_iou"] = L.smooth_l1_loss(prb_iou, vlm_iou)

        return losses


class WeightedApproximateVLMLoss:
    """
    Computes (1:1) losses between the predictions of linear probe
    and the predictions of VLM.

    Goal is to see if an intermediate layer can approximate, the
    outputs of the final layers including the false positives.

    Why? To see what layers are absolutely useful.
    """

    def __init__(self, args):
        self.device = DEVICE
        self.probe_mode = args.probe_mode
        self.num_spatial_units = int(args.num_spatial_units)
        self.num_classes = len(MC.get(args.labelset).thing_classes)
        self.use_classified_boxes = bool(args.use_classified_boxes)

    def __call__(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_feats_outputs: Dict[str, List[Tensor]],
        batched_probe_outputs: Dict[str, List[Tensor]],
    ) -> Dict[str, Any]:

        losses = {}

        if "pred_box" in batched_probe_outputs:
            im_sizes = [(bi["height"], bi["width"]) for bi in batched_inputs]

            # Step 1: Format dt/gt boxes and classes.
            batched_prb_box = [pb for pb in batched_probe_outputs["pred_box"]]
            batched_vlm_box = normalize_boxes(
                [bf for bf in batched_feats_outputs["pred_box"]],
                im_sizes,
                to="rel",
            )

            if self.use_classified_boxes:
                batched_prb_box = [pb.softmax(dim=-1) for pb in batched_prb_box]
                batched_vlm_box = [
                    F.one_hot(
                        (box * self.num_spatial_units).long(),
                        self.num_spatial_units,
                    ).float()
                    for box in batched_vlm_box
                ]
            else:
                batched_prb_box = [pb.sigmoid() for pb in batched_prb_box]

            # Step 2: Flatten preds and targs.
            prb_box = torch.cat(batched_prb_box, dim=0)
            vlm_box = torch.cat(batched_vlm_box, dim=0)

            # Step 3: Compute losses for boxes.
            if self.use_classified_boxes:
                losses["loss_box"] = L.binary_cross_entropy_with_logits(
                    prb_box, vlm_box
                )
            else:
                losses["loss_box"] = L.smooth_l1_loss(prb_box, vlm_box)

        if "pred_cls" in batched_probe_outputs:

            # Step 1: Format dt/gt boxes and classes.
            batched_prb_cls = [pc for pc in batched_probe_outputs["pred_cls"]]
            batched_vlm_cls = [
                F.one_hot(pc.long(), self.num_classes).float()
                for pc in batched_feats_outputs["pred_cls"]
            ]

            # Step 2: Flatten preds and targs.
            prb_cls = torch.cat(batched_prb_cls, dim=0)
            vlm_cls = torch.cat(batched_vlm_cls, dim=0)

            # Step 3: Compute losses for class.
            losses["loss_cls"] = L.binary_cross_entropy_with_logits(prb_cls, vlm_cls)

        return losses


class StandardLoss:
    """
    Computes losses between the predictions of linear probe
    and the ground truths via bipartite matching.

    NOTE: Unlike PredToPredLosses num_preds != num_trues.

    Standard training for DETR architectures.
    """

    def __init__(self, args):
        self.device = DEVICE
        self.num_spatial_units = int(args.num_spatial_units)
        self.bg_cls_weight = 2.0
        self.use_classified_boxes = bool(args.use_classified_boxes)

    def format_boxes(
        self,
        raw_pred_box: List[Tensor],
        raw_true_box: List[Tensor],
        im_sizes: List[Tuple[int, int]],
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
    ]:
        abs_true_box = raw_true_box
        rel_true_box = normalize_boxes(abs_true_box, im_sizes, to="rel")
        cls_true_box = [
            F.one_hot(
                torch.floor(box * self.num_spatial_units).long(),
                self.num_spatial_units,
            )
            for box in rel_true_box
        ]

        if self.use_classified_boxes:
            cls_pred_box = raw_pred_box
            rel_pred_box = [
                torch.div(box.argmax(dim=-1), self.num_spatial_units)
                for box in cls_pred_box
            ]
        else:
            rel_pred_box = raw_pred_box
            cls_pred_box = [
                F.one_hot(
                    (box * self.num_spatial_units).long(),
                    self.num_spatial_units,
                )
                for box in rel_pred_box
            ]

        abs_pred_box = normalize_boxes(rel_pred_box, im_sizes, to="abs")

        return (
            abs_pred_box,
            abs_true_box,
            rel_pred_box,
            rel_true_box,
            cls_pred_box,
            cls_true_box,
        )

    def __call__(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_feats_outputs: Dict[str, List[Tensor]],
        batched_probe_outputs: Dict[str, List[Tensor]],
    ):
        im_sizes = [(bi["height"], bi["width"]) for bi in batched_inputs]

        # Step 1: Format boxes and classes.
        raw_pred_box = [pb for pb in batched_probe_outputs["pred_box"]]
        raw_true_box = [bi["instances"].gt_boxes.tensor for bi in batched_inputs]

        (
            abs_pred_box,
            abs_true_box,
            rel_pred_box,
            rel_true_box,
            cls_pred_box,
            cls_true_box,
        ) = self.format_boxes(raw_pred_box, raw_true_box, im_sizes)

        true_cls = [bi["instances"].gt_classes for bi in batched_inputs]
        pred_cls = [pc for pc in batched_outputs["pred_cls"]]

        # Step 2: Compute match costs.
        if self.use_classified_boxes:
            batched_match_cost_box = [
                L.binary_cross_entropy_with_logits(pred_box, true_box, reduction="none")
                for pred_box, true_box in zip(cls_pred_box, cls_true_box)
            ]
        else:
            batched_match_cost_box = [
                torch.cdist(pred_box, true_box, p=1)
                for pred_box, true_box in zip(rel_pred_box, rel_true_box)
            ]

        batched_match_cost_iou = [
            generalized_box_iou_loss(pred_box, true_box, reduction="none")
            for pred_box, true_box in zip(abs_pred_box, abs_true_box)
        ]

        batched_match_cost_cls = [
            sigmoid_focal_loss(pred_cls, true_cls, reduction="none")
            for pred_cls, true_cls in zip(pred_cls, true_cls)
        ]

        batched_match_cost = [
            mc_iou + mc_box + mc_cls
            for mc_iou, mc_box, mc_cls in zip(
                batched_match_cost_iou,
                batched_match_cost_box,
                batched_match_cost_cls,
            )
        ]

        # Step 3: Compute no. of preds for each pred.
        batched_n_preds = [len(p) for p in abs_pred_box]

        # Step 4: Populate true boxes and and match them to pred boxes.
        (
            abs_true_box,
            true_cls,
            box_weights,
            cls_weights,
            n_positives,
            n_negatives,
        ) = self.match_boxes(
            abs_pred_box, abs_true_box, batched_match_cost, batched_n_preds
        )

        # Step 5: Flatten boxes across samples.
        abs_pred_box = torch.cat(abs_pred_box, dim=0)
        abs_true_box = torch.cat(abs_true_box, dim=0)

        rel_pred_box = torch.cat(rel_pred_box, dim=0)
        rel_true_box = torch.cat(rel_true_box, dim=0)

        cls_pred_box = torch.cat(cls_pred_box, dim=0)
        cls_true_box = torch.cat(cls_true_box, dim=0)

        pred_cls = torch.stack(pred_cls)
        true_cls = torch.stack(true_cls)

        # Step 6: Compute mean factors for batch.
        mean_factor_box = float(sum(n_positives))
        mean_factor_cls = float(
            sum(
                n_pos * 1.0 + self.bg_cls_weight * n_neg
                for n_pos, n_neg in zip(n_positives, n_negatives)
            )
        )

        # Step 7: Compute losses.
        if self.use_classified_boxes:
            loss_box = (
                L.binary_cross_entropy_with_logits(
                    cls_pred_box, cls_true_box, reduction="sum"
                )
                * box_weights
                / (mean_factor_box + self.eps)
            )
        else:
            loss_box = (
                L.smooth_l1_loss(rel_pred_box, rel_true_box, reduction="sum")
                * box_weights
                / (mean_factor_box + self.eps)
            )

        loss_iou = (
            generalized_box_iou_loss(abs_pred_box, abs_true_box, reduction="sum")
            * box_weights
            / (mean_factor_box + self.eps)
        )

        loss_cls = (
            sigmoid_focal_loss(pred_cls, true_cls, reduction="sum")
            * cls_weights
            / (mean_factor_cls + self.eps)
        )

        return {
            "loss_box": loss_box,
            "loss_iou": loss_iou,
            "loss_cls": loss_cls,
        }

    def match_boxes(
        self,
        batched_true_box: List[Tensor],
        batched_true_cls: List[Tensor],
        batched_match_cost: List[Tensor],
        batched_n_preds: List[int],
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[int], List[int]
    ]:
        """
        Matches predicted boxes and classes to corresponding
        ground truth boxes and populates true_box and true_cls
        to match len(pred_box). Expects all boxes to be in
        absolute (xyxy) format.

        Returns:
        true_box and true_cls with len(.) == len(pred_box)
        """

        batched_new_true_box, batched_new_true_cls = [], []
        batched_box_weights, batched_cls_weights = [], []
        batched_num_positives, batched_num_negatives = [], []

        for true_box, true_cls, match_cost, n_preds in zip(
            batched_true_box, batched_true_cls, batched_match_cost, batched_n_preds
        ):
            n_trues = len(true_box)

            # initially match everything to background and only update positives.
            matched_true_box = torch.full(
                (n_preds, 4), 0, dtype=torch.long, device=self.device
            )
            matched_true_ind = torch.full(
                (n_preds,), 0, dtype=torch.long, device=self.device
            )
            matched_true_cls = torch.full(
                (n_preds,), 0, dtype=torch.long, device=self.device
            )

            if n_preds == 0 or len(true_box) == 0:
                batched_new_true_box.append(matched_true_box)
                batched_new_true_cls.append(matched_true_cls)

                batched_num_positives.append(n_trues)
                batched_num_negatives.append(n_preds - n_trues)

                batched_box_weights.append(matched_true_ind)
                batched_cls_weights.append(matched_true_ind)

                continue

            match_row_inds, match_col_inds = linear_sum_assignment(match_cost)

            match_row_inds = torch.from_numpy(match_row_inds).to(self.device)
            match_col_inds = torch.from_numpy(match_col_inds).to(self.device)

            matched_true_ind[match_row_inds] = match_col_inds + 1
            matched_true_cls[match_row_inds] = true_cls[match_col_inds]
            matched_true_box[matched_true_ind] = true_box

            batched_new_true_box.append(matched_true_box)
            batched_new_true_cls.append(matched_true_cls)

        return (
            batched_new_true_box,
            batched_new_true_cls,
            batched_box_weights,
            batched_cls_weights,
            batched_num_positives,
            batched_num_negatives,
        )
