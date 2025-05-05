# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict
from torch import Tensor

from .base_aligner import Aligner
from ..schema import MinimalSchemaFrame
from ..accessors import Recorder
from ..types import IOFrame


class GENUAligner(Aligner):
    def _process_det_decoder_resid_subpattern(
        self, infer_id: int, subpattern: str, feature: Tensor
    ) -> List[MinimalSchemaFrame]:
        match subpattern:
            case "pre_img" | "pos_img" | "refpnts":
                return [
                    MinimalSchemaFrame(
                        **{
                            "image_id": batch_id,
                            "infer_id": infer_id,
                            "heads_id": -1,
                            "token_id": token_id,
                            "features": tensor.detach().cpu().tolist(),
                        }
                    )
                    for batch_id, tokens in enumerate(feature.permute(0, 1, 2))
                    for token_id, tensor in enumerate(tokens)
                ]
            case "pre_txt" | "pos_txt":
                return [
                    MinimalSchemaFrame(
                        **{
                            "image_id": batch_id,
                            "infer_id": infer_id,
                            "heads_id": -1,
                            "token_id": token_id,
                            "features": tensor.detach().cpu().tolist(),
                        }
                    )
                    for batch_id, tokens in enumerate(feature)
                    for token_id, tensor in enumerate(tokens)
                ]
            case unsupported:
                raise NotImplementedError(f"unsupported sub_pattern: {unsupported}.")

    def process_det_decoder_resid(
        self, recorder: Recorder
    ) -> Dict[str, List[MinimalSchemaFrame]]:
        return self._decompose_with(
            self._process_det_decoder_resid_subpattern, recorder
        )

    def _process_det_decoder_mha_subpattern(
        self, infer_id: int, subpattern: str, feature: Tensor
    ) -> List[MinimalSchemaFrame]:
        match subpattern:
            case "weights":
                return [
                    MinimalSchemaFrame(
                        **{
                            "image_id": batch_id,
                            "infer_id": infer_id,
                            "heads_id": heads_id,
                            "token_id": token_id,
                            "features": tensor.detach().cpu().tolist(),
                        }
                    )
                    for batch_id, attn_heads in enumerate(feature)
                    for heads_id, attn_head_weight in enumerate(attn_heads)
                    for token_id, tensor in enumerate(attn_head_weight)
                ]
            case "outputs":
                return [
                    MinimalSchemaFrame(
                        **{
                            "image_id": batch_id,
                            "infer_id": infer_id,
                            "heads_id": -1,
                            "token_id": token_id,
                            "features": tensor.detach().cpu().tolist(),
                        }
                    )
                    for batch_id, tokens in enumerate(feature.permute(1, 0, 2))
                    for token_id, tensor in enumerate(tokens)
                ]
            case unsupported:
                raise NotImplementedError(f"unsupported sub_pattern: {unsupported}.")

    def process_det_decoder_mha(
        self, recorder: Recorder
    ) -> Dict[str, List[MinimalSchemaFrame]]:
        return self._decompose_with(self._process_det_decoder_mha_subpattern, recorder)

    def _process_det_decoder_mlp_subpattern(
        self, infer_id: int, subpattern: str, feature: Tensor
    ) -> List[MinimalSchemaFrame]:
        match subpattern:
            case "pos_img":
                return [
                    MinimalSchemaFrame(
                        **{
                            "image_id": batch_id,
                            "infer_id": infer_id,
                            "heads_id": -1,
                            "token_id": token_id,
                            "features": tensor.detach().cpu().tolist(),
                        }
                    )
                    for batch_id, tokens in enumerate(feature.permute(0, 1, 2))
                    for token_id, tensor in enumerate(tokens)
                ]
            case unsupported:
                raise NotImplementedError(f"unsupported sub_pattern: {unsupported}.")

    def process_det_decoder_mlp(
        self, recorder: Recorder
    ) -> Dict[str, List[MinimalSchemaFrame]]:
        return self._decompose_with(self._process_det_decoder_mlp_subpattern, recorder)

    def process_io(self, recorder: Recorder) -> IOFrame:
        records = recorder.fetch_records()["io"][0]
        assert "batched_inputs" in records, "require batched_inputs."

        bis = records["batched_inputs"]
        sio = IOFrame(
            image_id=[bi["image_id"] for bi in bis],
            filename=[str(bi["file_name"]) for bi in bis],
            true_box=[bi["instances"].gt_boxes.tensor for bi in bis],
            true_cls=[bi["instances"].gt_classes for bi in bis],
        )

        if "batched_outputs" in records:
            bos = records["batched_outputs"]
            sio.pred_box = [bo["unprocessed_boxes"].tensor for bo in bos]
            sio.pred_cls = [bo["unprocessed_clsid"] for bo in bos]
            sio.pred_scr = [bo["unprocessed_score"] for bo in bos]

        if "confusion_mask" in records:
            sio.conf_msk = records["confusion_mask"]

        return sio
