# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from copy import deepcopy
from blueglass.utils.logger_utils import setup_blueglass_logger
import torch
from blueglass.third_party.detectron2.data import MetadataCatalog
from blueglass.third_party.detectron2.evaluation import DatasetEvaluator
from blueglass.third_party.detectron2.structures import Instances, BoxMode
from blueglass.visualize.detections import visualize_detections
from blueglass.configs import BLUEGLASSConf, Datasets
from .matchers import prepare_matcher


logger = setup_blueglass_logger(__name__)


def prepare_bbox_format(ds: Datasets):
    return BoxMode.XYXY_ABS


class LabelMatchingEvaluator(DatasetEvaluator):
    def __init__(
        self, conf: BLUEGLASSConf, subeval: DatasetEvaluator, runner_mode: str = None
    ):
        assert isinstance(
            subeval, DatasetEvaluator
        ), "invalid evaluator type for subeval."

        self.device = torch.device(
            "cuda:0" if torch.cuda.device_count() > 0 else "cuda"
        )
        self.conf = conf
        conf_dataset = (
            conf.dataset.infer if runner_mode == "infer" else conf.dataset.test
        )

        self.num_vis_samples = conf.evaluator.num_vis_samples
        self.dt_bbox_format = BoxMode.XYXY_ABS
        self.gt_bbox_format = prepare_bbox_format(conf_dataset)
        self.output_dir = conf.experiment.output_dir
        self.vis_save_dpath = os.path.join(self.output_dir, "vis")

        if not os.path.exists(self.vis_save_dpath):
            os.makedirs(self.vis_save_dpath)

        self.classnames = MetadataCatalog.get(conf.dataset.label).thing_classes
        self.num_topk_matches = min(
            self.conf.evaluator.num_topk_matches, len(self.classnames)
        )
        self.matcher = prepare_matcher(
            conf.evaluator.matcher,
            self.classnames,
            conf.evaluator.prompter,
            conf.evaluator.encoder,
            conf.evaluator.use_negatives,
            conf.evaluator.use_parts,
            conf.evaluator.num_topk_matches,
        )
        self.subeval = subeval

    def reset(self):
        return self.subeval.reset()

    def process(self, inputs, outputs, sub_eval=True):
        outputs = deepcopy(outputs)
        for inp, out in zip(inputs, outputs):
            if len(out["instances"]) == 0:
                continue

            inst = out["instances"]
            inst = self.transform_bbox_format(inst)

            if self.conf.evaluator.use_descriptions:
                inst = self.transform_descriptions_and_scores(inst)

            inst = self.filter_ood_predictions(inst)
            inst = self.filter_by_score(inst)
            inst = self.filter_by_count(inst)

            out["instances"] = inst

        self.visualize_samples(inputs, outputs)
        if sub_eval:
            self.subeval.process(inputs, outputs)
        return outputs

    def evaluate(self):
        return self.subeval.evaluate()

    def transform_bbox_format(self, inst: Instances) -> Instances:
        assert isinstance(inst, Instances), "unsupported data format for instances."
        assert inst.has("pred_boxes"), "missing pred_boxes in instances."

        inst.pred_boxes = BoxMode.convert(
            inst.pred_boxes,
            self.dt_bbox_format,
            self.gt_bbox_format,
        )
        return inst

    def transform_descriptions_and_scores(self, inst: Instances) -> Instances:
        assert isinstance(inst, Instances), "unsupported data format for instances."
        assert inst.has(
            "pred_object_descriptions"
        ), "missing descriptions in instances."

        cids, scores = self.matcher.to_cids_and_scores(
            inst.pred_object_descriptions.data
        )
        cids, scores = cids.to(self.device), scores.to(self.device)

        for attr, flag, label in [
            ("pred_box_ious", self.conf.evaluator.use_box_ious, "IOU"),
            ("pred_box_objectness", self.conf.evaluator.use_box_objectness, "objectness"),
        ]:
            if flag:
                val = getattr(inst, attr, None)
                if inst.has(attr) and val is not None and val.ndim == 1:
                    scores *= val.unsqueeze(-1).repeat(1, self.num_topk_matches)
                else:
                    logger.warning(f"Skipping box {label} weighting: '{attr}' missing or malformed.")

        sorted_scores, sorted_indices = scores.flatten().sort(descending=True)

        unflattened_ind = torch.div(
            sorted_indices, self.num_topk_matches, rounding_mode="floor"
        )

        new_inst = Instances(inst.image_size)
        new_inst.pred_boxes = inst.pred_boxes[unflattened_ind]
        new_inst.pred_object_descriptions = inst.pred_object_descriptions[
            unflattened_ind
        ]
        new_inst.pred_classes = cids.flatten()[sorted_indices]
        new_inst.scores = sorted_scores

        # Copy over optional fields if enabled and available
        for attr, flag in [
            ("pred_box_ious", self.conf.evaluator.use_box_ious),
            ("pred_box_objectness", self.conf.evaluator.use_box_objectness),
        ]:
            if flag and inst.has(attr):
                setattr(new_inst, attr, getattr(inst, attr)[unflattened_ind])

        return new_inst

    def filter_ood_predictions(self, inst: Instances) -> Instances:
        assert isinstance(inst, Instances), "unsupported data format for instances."
        assert inst.has("pred_classes"), "missing pred_classes in instances."
        return inst[inst.pred_classes != -1]

    def filter_by_score(self, inst: Instances) -> Instances:
        assert isinstance(inst, Instances), "unsupported data format for instances."
        assert inst.has("scores"), "missing scores in instances."
        return inst[inst.scores > self.conf.evaluator.min_threshold_cls]

    def filter_by_count(self, inst: Instances) -> Instances:
        assert isinstance(inst, Instances), "unsupported data format for instances."
        assert inst.has("scores"), "missing scores in instances."
        return inst[
            inst.scores.topk(
                min(len(inst), self.conf.evaluator.max_predictions)
            ).indices
        ]

    def visualize_samples(self, inputs, outputs):
        if self.num_vis_samples > 0 and len(inputs) > 0:
            inp, out = inputs[0], outputs[0]

            bboxes = out["instances"].pred_boxes.tensor.cpu().numpy()
            scores = out["instances"].scores.cpu().tolist()
            labels = out["instances"].pred_classes.cpu().tolist()

            visualize_detections(
                os.path.join(self.vis_save_dpath, f"{inp['image_id']}.jpg"),
                image=inp["file_name"],
                bboxes=bboxes,
                cnames=[self.classnames[c] for c in labels],
                scores=scores,
            )
            self.num_vis_sample -= 1
