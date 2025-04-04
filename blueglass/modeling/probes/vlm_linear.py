# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict, Any, Literal, Optional, Union
import torch
from torch import nn, Tensor, inference_mode
from blueglass.third_party.detectron2.data import MetadataCatalog
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.evaluation import compute_confusion_mask
from blueglass.modeling.build import build_model
from .losses import ApproximateVLMLoss, StandardLoss, normalize_boxes


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_data() -> List[Dict[str, Any]]:
    import cv2

    im_path = "./assets/sample.jpg"
    im = torch.from_numpy(cv2.imread(im_path)).permute(2, 0, 1)
    im_h, im_w = im.shape[1:]
    return [
        {
            "file_name": im_path,
            "image_id": "000",
            "height": im_h,
            "width": im_w,
            "image": im,
            "instances": Instances((im_h, im_w)),
        }
    ]


def inverse_sigmoid(tensor: Tensor, eps=1e-3):
    tensor = tensor.clamp(min=0, max=1)
    numera = tensor.clamp(min=eps)
    denomi = (1 - tensor).clamp(min=eps)
    return torch.log(numera / denomi)


class BoxProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = DEVICE
        self.box_deltas = nn.Linear(in_dim, out_dim).to(self.device)

    def forward(self, batched_features: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        deltas = self.box_deltas(
            batched_features[FeatureSubPatterns.POS_IMG].to(self.device)
        )
        refpts = inverse_sigmoid(
            batched_features[FeatureSubPatterns.REFPNTS].to(self.device)
        )
        pred_box = refpts + deltas
        return {"pred_box": [pb for pb in pred_box]}


class ClsProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = DEVICE
        self.cls_scores = nn.Linear(in_dim, out_dim).to(self.device)

    def forward(self, batched_features: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        pred_cls = self.cls_scores(
            batched_features[FeatureSubPatterns.POS_IMG].to(self.device)
        )
        return {"pred_cls": [pc for pc in pred_cls]}


class IoUProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.iou_scores = nn.Linear(in_dim, out_dim)

    def forward(self, batched_features: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        pred_iou = self.iou_scores(batched_features[FeatureSubPatterns.POS_IMG])
        return {"pred_iou": [pi for pi in pred_iou]}


class DetProbe(nn.Module):
    """
    Detection probe layer for localization and classification.
    """

    def __init__(
        self, in_dim: int, box_out_dim: int, cls_out_dim: int, iou_out_dim: int = 1
    ):
        super().__init__()
        self.box_deltas = nn.Linear(in_dim, box_out_dim)
        self.cls_scores = nn.Linear(in_dim, cls_out_dim)
        self.iou_scores = nn.Linear(in_dim, iou_out_dim)

    def forward(
        self, batched_features: Dict[str, List[Tensor]]
    ) -> Dict[str, List[Tensor]]:
        bfeatures = torch.stack(batched_features["pos_features_img"])
        refpoints = torch.stack(batched_features["refpoint"])
        assert refpoints.shape[-1] == 4, "expected refpoints to be 4x."
        pred_box = self.box_deltas(bfeatures) + inverse_sigmoid(refpoints)
        pred_cls = self.cls_scores(bfeatures)
        pred_iou = self.iou_scores(bfeatures)
        return {
            "pred_box": [pb for pb in pred_box],
            "pred_cls": [pc for pc in pred_cls],
            "pred_iou": [pi for pi in pred_iou],
        }


class DropoutLinearProbe(nn.Module):
    """
    Linear probe with Dropouts,
    allowing to compute prediction (epistemic)
    uncertainities during inference using MC Dropout.
    """

    def __init__(
        self,
        in_dim: int,
        box_out_dim: int,
        cls_out_dim: int,
        mc_dropout_pct: float = 0.5,
        mc_infer_steps: int = 10,
    ):
        super().__init__()
        self.mc_dropout_pct = mc_dropout_pct
        self.mc_infer_steps = mc_infer_steps

        self.box_dropped_linear = nn.Sequential(
            nn.Dropout(self.dropout_pct), nn.Linear(in_dim, box_out_dim)
        )
        self.cls_dropped_linear = nn.Sequential(
            nn.Dropout(self.dropout_pct), nn.Linear(in_dim, cls_out_dim)
        )

    def forward_once(self, features) -> Dict[str, Tensor]:
        return {
            "pred_box": self.box_dropped_linear(features),
            "pred_cls": self.cls_dropped_linear(features),
        }

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        if self.training:
            return self.forward_once(features)
        else:
            mc_preds = [self.forward_once(features) for _ in range(self.mc_infer_steps)]
            pred_box = torch.cat([p["pred_box"] for p in mc_preds], dim=0)
            pred_cls = torch.cat([p["pred_cls"] for p in mc_preds], dim=0)

            return {
                "pred_box": pred_box.mean(dim=0),
                "pred_cls": pred_cls.mean(dim=0),
                "uvar_box": pred_box.var(dim=0),
                "uvar_cls": pred_cls.var(dim=0),
            }


class FeatureSplitProbe(nn.Module):
    """
    Splits the feature into two parts, "semantic" and "spatial"
    and uses the corresponding part to compute boxes or classes.

    Allows to isolate features corresponding to each representation.
    """

    def __init__(
        self, in_dim: int, box_out_dim: int, cls_out_dim: int, subfeature_dim: int
    ):
        super().__init__()
        self.sub_linear = nn.Linear(in_dim, 2 * subfeature_dim)
        self.box_linear = nn.Linear(subfeature_dim, box_out_dim)
        self.cls_linear = nn.Linear(subfeature_dim, cls_out_dim)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        box_sf, cls_sf = torch.split(self.sub_linear(features), 2)
        return {
            "pred_box": self.box_linear(box_sf),
            "pred_cls": self.cls_linear(cls_sf),
        }


class LinearProbedVLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = DEVICE

        self.target_vlm = build_model(args).eval().to(self.device)
        self.freeze_vlm()

        self.preprocess_seqs = build_sequence_processor(args)
        self.feature_pattern = FeaturePatterns(args.feature_pattern)
        self.feature_subpatn = FeatureSubPatterns(args.feature_subpattern)
        self.use_classified_boxes = args.use_classified_boxes

        sm.register(self.feature_pattern)

        self.losses_per_layer = (
            ApproximateVLMLoss(args)
            if args.use_vlm_pred_as_true
            else StandardLoss(args)
        )
        self.probes = self.build_probes(args).to(self.device)

    def freeze_vlm(self):
        for p in self.target_vlm.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        self.probes = self.probes.train(mode)
        return self

    def infer_in_dims(self, args) -> Dict[str, int]:
        assert sm.is_registered(self.feature_pattern), "pattern not registered."

        sm(self.feature_pattern).new()

        with inference_mode():
            assert not self.target_vlm.training, "target vlm set to train."
            self.target_vlm(infer_data())

        return {
            layer: values[self.feature_subpatn.value].shape[-1]
            for layer, values in self.extract_features()
            .fetch(self.feature_pattern, self.feature_subpatn)
            .items()
        }

    def build_probes(self, args) -> nn.ModuleDict:
        if args.use_classified_boxes:
            box_out_dim = 4 * args.num_spatial_units
        else:
            box_out_dim = 4

        cls_out_dim = len(MetadataCatalog.get(args.labelset).thing_classes)
        prb_in_dims = self.infer_in_dims(args)

        if args.probe_mode == "box_linear":
            return nn.ModuleDict(
                {
                    layer_name: BoxProbe(in_dim, box_out_dim)
                    for layer_name, in_dim in prb_in_dims.items()
                }
            )

        if args.probe_mode == "cls_linear":
            return nn.ModuleDict(
                {
                    layer_name: ClsProbe(in_dim, cls_out_dim)
                    for layer_name, in_dim in prb_in_dims.items()
                }
            )

        if args.probe_mode == "det_linear":
            return nn.ModuleDict(
                {
                    layer_name: DetProbe(in_dim, box_out_dim, cls_out_dim)
                    for layer_name, in_dim in prb_in_dims.items()
                }
            )

        if args.probe_mode == "det_dropout":
            return nn.ModuleDict(
                {
                    layer_name: DropoutLinearProbe(
                        in_dim,
                        box_out_dim,
                        cls_out_dim,
                        args.mc_dropout_pct,
                        args.mc_infer_steps,
                    )
                    for layer_name, in_dim in prb_in_dims.items()
                }
            )

        if args.probe_mode == "det_feature_split":
            return nn.ModuleDict(
                {
                    layer_name: FeatureSplitProbe(
                        in_dim, box_out_dim, cls_out_dim, args.subfeature_dim
                    )
                    for layer_name, in_dim in prb_in_dims.items()
                }
            )

        raise ValueError("unsupported probe mode.")

    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        branch: Optional[Literal["extraction", "fwd_probes"]] = None,
    ) -> Union[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        if branch is None:
            assert not self.training, "branch==None means infer mode."
            return self.inference(batched_inputs)

        batched_vlm_outputs = self.forward_vlm(batched_inputs)

        if branch == "extraction":
            return batched_vlm_outputs

        if branch == "fwd_probes":
            return self.forward_probe(batched_inputs, batched_vlm_outputs)

        raise Exception("invalid branch name.")

    def inference(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        # Step 1: extract predictions and features from vlm.
        batched_vlm_outputs = self.forward_vlm(batched_inputs)

        # Step 2: extract predictions from probes.
        batched_prb_outputs = self.forward_probe(batched_inputs, batched_vlm_outputs)

        # Step 3: merge predictions.
        return self.merge_predictions(
            batched_inputs, batched_vlm_outputs, batched_prb_outputs
        )

    def forward_vlm(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        sm(self.feature_pattern).new()

        assert not self.target_vlm.training, "target vlm in train mode."

        with inference_mode():
            batched_outputs = self.target_vlm(batched_inputs)

        sm(self.feature_pattern).record(
            "io",
            {
                "batched_inputs": batched_inputs,
                "batched_outputs": batched_outputs,
            },
        )
        return {
            "batched_outputs": batched_outputs,
            "batched_features": self.extract_features(),
        }

    def forward_probe(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_vlm_outputs: List[Dict[str, Any]],
    ) -> Union[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:

        # Step 1: extract intermediate features.
        features = batched_vlm_outputs["batched_features"].fetch(self.feature_pattern)

        # Step 2: forward features through probes.
        probe_outputs = {
            layer_name: probe(features[layer_name])
            for layer_name, probe in self.probes.items()
        }

        # Step 3: compute loss or return instances.
        if self.training:
            return {
                layer_name: self.losses_per_layer(
                    batched_inputs, batched_vlm_outputs["batched_features"], probe_out
                )
                for layer_name, probe_out in probe_outputs.items()
            }
        else:
            return {
                layer_name: self.postprocess_per_layer(
                    batched_inputs, batched_vlm_outputs["batched_features"], probe_out
                )
                for layer_name, probe_out in probe_outputs.items()
            }

    def postprocess_per_layer(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_feats_outputs,
        batched_probe_outputs: Dict[str, List[Tensor]],
    ) -> List[Dict[str, Any]]:
        im_sizes = [(bi["height"], bi["width"]) for bi in batched_inputs]

        if "pred_box" in batched_probe_outputs:
            if self.use_classified_boxes:
                pred_box = [
                    box.softmax(dim=-1).argmax(dim=-1)
                    for box in batched_probe_outputs["pred_box"]
                ]
            else:
                pred_box = [box.sigmoid() for box in batched_probe_outputs["pred_box"]]

            pred_box = normalize_boxes(pred_box, im_sizes, to="abs")
        else:
            io = batched_feats_outputs.fetch_io(IOPatterns.PRED_BOX)
            assert "pred_box" in io, "boxes not found in probe or features outputs."

            pred_box = [box for box in io["pred_box"]]

        if "pred_cls" in batched_probe_outputs:
            pred_max = [
                pcls.sigmoid().max(dim=-1) for pcls in batched_probe_outputs["pred_cls"]
            ]
            pred_scr = [mx.values for mx in pred_max]
            pred_cls = [mx.indices for mx in pred_max]
        else:
            io = batched_feats_outputs.fetch_io(IOPatterns.PRED)
            assert "pred_scr" in io, "scores not found in probe or vlm outputs."
            assert "pred_cls" in io, "clases not found in probe or vlm outputs."

            pred_scr = [scr for scr in io["pred_scr"]]
            pred_cls = [cls for cls in io["pred_cls"]]

        return [
            {
                "instances": Instances(
                    sz, pred_boxes=Boxes(pb), pred_classes=pc, scores=sc
                )
            }
            for sz, pb, pc, sc in zip(im_sizes, pred_box, pred_cls, pred_scr)
        ]

    def extract_features(self) -> Any:
        return self.preprocess_seqs(sm(self.feature_pattern).end())

    def merge_predictions(
        self,
        batched_inputs: List[Dict[str, Any]],
        batched_vlm_outputs: Dict[str, Union[List[Dict[str, Any]], Any]],
        batched_prb_outputs: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        assert (
            "batched_features" in batched_vlm_outputs
        ), "expected feature outputs in vlm outputs."

        batched_features = batched_vlm_outputs["batched_features"].fetch_io(
            IOPatterns.PRED
        )

        assert isinstance(
            batched_features, Dict
        ), "expected features to be dict of io patterns."

        imszs = [(bi["height"], bi["width"]) for bi in batched_inputs]

        insts = [
            {
                "instances": Instances(
                    sz, pred_boxes=Boxes(bx), pred_classes=cs, scores=sc
                )
            }
            for sz, bx, cs, sc in zip(
                imszs,
                batched_features["pred_box"],
                batched_features["pred_cls"],
                batched_features["pred_scr"],
            )
        ]

        return {"vlm_unprocessed": insts, **batched_prb_outputs}
