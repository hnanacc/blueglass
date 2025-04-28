# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from collections import defaultdict
from blueglass.utils.logger_utils import setup_blueglass_logger
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from torch import nn
from torchvision import transforms as T
from blueglass.third_party.detectron2.data import MetadataCatalog as MC
from blueglass.third_party.detectron2.structures import Instances, Boxes, BoxMode
from blueglass.modeling.modelstore.grounding_dino.groundingdino.util.vl_utils import (
    build_captions_and_token_span,
    create_positive_map_from_span,
)
from blueglass.evaluation import compute_confusion_mask
from blueglass.configs import BLUEGLASSConf, FeaturePattern
from blueglass.features import intercept_manager

logger = setup_blueglass_logger("blueglass")


class GroundingDINO(nn.Module):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__()
        from blueglass.modeling.modelstore.grounding_dino.groundingdino.util.inference import (
            load_model,
        )

        self.conf = conf

        assert conf.model.conf_path is not None, "missing conf_path for Grounding DINO."
        assert (
            conf.model.checkpoint_path is not None
        ), "missing checkpoint_path for Grounding DINO."

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(
            conf.model.conf_path,
            conf.model.checkpoint_path,
        ).to(self.device)

        self.box_threshold = conf.evaluator.min_threshold_box
        self.txt_threshold = conf.evaluator.min_threshold_cls

        self.classnames = MC.get(conf.dataset.label).thing_classes
        self.chunk_size = min(self.infer_chunk_size(conf), len(self.classnames))
        self.chunked_cnames = self.prepare_chunked_cnames(self.classnames)

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(800, max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def infer_chunk_size(self, conf: BLUEGLASSConf):
        match conf.dataset.label.split("_")[0].lower():
            case "lvis":
                return 72
            case "o365":
                return 80
            case _:
                return 96

    def prepare_chunked_cnames(self, classnames: List[str]):
        num_cnames = len(self.classnames)
        start_idx, chunked_cnames = 0, []
        while start_idx < num_cnames:
            end_idx = min(num_cnames, start_idx + self.chunk_size)
            cur_chunk = classnames[start_idx:end_idx]
            chunked_cnames.append(cur_chunk)
            start_idx += len(cur_chunk)

        assert sum(len(c) for c in chunked_cnames) == len(
            self.classnames
        ), "num chunked classes are less than actual classes."

        return chunked_cnames

    def prepare_caption_and_token_mask(self, cnames: List[str]):
        caption, token_span_dict, modified_cnames = build_captions_and_token_span(
            cnames, True, True
        )
        token_span = [token_span_dict[mc] for mc in modified_cnames]
        cap_tokens = self.model.tokenizer(caption)
        token_mask = create_positive_map_from_span(cap_tokens, token_span)

        assert len(cap_tokens.input_ids) < 256, "tokens exceed the context size."
        assert len(cnames) == len(
            modified_cnames
        ), "there were some classnames removed during caption computation."

        return caption, token_mask

    @torch.inference_mode()
    def forward(self, batched_inputs: List[Dict[str, Any]]):
        images = self.preprocess(batched_inputs)

        chunked_outputs = defaultdict(list)
        for cnames in self.chunked_cnames:
            # combines cnames into a caption and creates a token mask
            # corresponding to the position of tokens of each class
            # in the caption. token_mask.shape = (num_classes, context_size).
            caption, token_mask = self.prepare_caption_and_token_mask(cnames)

            outputs = self.model(
                images,
                captions=[caption] * len(batched_inputs),
                blueglass_conf=self.conf,
            )

            chunked_outputs["token_mask"].append(token_mask)
            chunked_outputs["pred_logit"].append(outputs["pred_logits"])
            chunked_outputs["pred_boxes"].append(outputs["pred_boxes"])

        concated_outputs = {
            k: torch.cat(chunks, dim=-2) for k, chunks in chunked_outputs.items()
        }

        # transform to probabilities and convert to prob_per_classname
        # by masking scores of tokens that do not belong to this class.
        # prob_per_token.shape = (batch_size, n_preds, context_size)
        prob_per_token = concated_outputs["pred_logit"].sigmoid()

        token_mask = concated_outputs["token_mask"].to(self.device)

        # prob_per_cname.shape = (batch_size, n_preds, n_classes)
        prob_per_cname = prob_per_token @ token_mask.T

        batch_size, num_preds, num_classes = prob_per_cname.shape

        # we find the bounding boxes with best scores. Note, we
        # match each box to all classes, and the pair which has
        # highest scores is selected. This means there are some
        # bbox predictions that could be removed and there will
        # be boxes that are repeated but assigned different classes.
        topk_score, topk_ind_flattened = torch.topk(
            prob_per_cname.reshape(batch_size, -1), num_preds, dim=-1
        )
        topk_cname = topk_ind_flattened % num_classes
        topk_bxind = topk_ind_flattened // num_classes
        topk_boxes = torch.gather(
            concated_outputs["pred_boxes"],
            1,
            topk_bxind.unsqueeze(-1).repeat(1, 1, 4),
        )

        batched_outputs = [
            {
                "pred_boxes": topk_boxes[bi],
                "pred_score": topk_score[bi],
                "pred_cname": topk_cname[bi],
                "unprocessed_boxes": concated_outputs["pred_boxes"][bi],
                "unprocessed_bxind": topk_bxind[bi],
                "unprocessed_clsid": prob_per_cname[bi].max(dim=1).indices,
                "unprocessed_score": prob_per_cname[bi].max(dim=1).values,
            }
            for bi in range(len(batched_inputs))
        ]

        return self.postprocess(batched_inputs, batched_outputs)

    def preprocess(self, batched_inputs: List[Dict]):
        return [self.transform(bi["image"]).to(self.device) for bi in batched_inputs]

    def postprocess(self, batched_inputs, batched_outputs):
        processed_batched_outputs = []
        for bi, bo in zip(batched_inputs, batched_outputs):
            new_inst = Instances((bi["height"], bi["width"]))
            new_inst.pred_boxes = Boxes(bo["pred_boxes"])
            new_inst.pred_boxes = BoxMode.convert(
                new_inst.pred_boxes, BoxMode.CXCYWH_ABS, BoxMode.XYXY_ABS
            )
            assert isinstance(new_inst.pred_boxes, Boxes), "unexpected conversion."
            new_inst.pred_boxes.scale(bi["width"], bi["height"])
            new_inst.scores = bo["pred_score"]
            new_inst.pred_classes = bo["pred_cname"]

            upd_boxes = Boxes(bo["unprocessed_boxes"])
            upd_boxes = BoxMode.convert(upd_boxes, BoxMode.CXCYWH_ABS, BoxMode.XYXY_ABS)
            assert isinstance(upd_boxes, Boxes), "unexpected conversion."
            upd_boxes.scale(bi["width"], bi["height"])

            processed_batched_outputs.append(
                {
                    "instances": new_inst,
                    "unprocessed_boxes": upd_boxes,
                    "unprocessed_bxind": bo["unprocessed_bxind"],
                    "unprocessed_clsid": bo["unprocessed_clsid"],
                    "unprocessed_score": bo["unprocessed_score"],
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
