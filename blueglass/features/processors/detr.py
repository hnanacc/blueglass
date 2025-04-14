# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from dataclasses import asdict
from typing import List, Dict, Tuple
import torch
from torch import Tensor

from .base_processor import Processor

from ..schema import MinimalSchemaFrame
from ..accessors import Recorder
from ..types import IOFrame


class DETRProcessor(Processor):
    def _expand_std_io(
        self,
        input_infer_id: int,
        total_infer_ids: int,
        std_io: IOFrame,
        num_heads: int = 1,
    ):
        """
        _expand_std_io adapted to gdino model

        Arranges tensors from a dictionary into rows, repeating the structure based on the number of attention heads.

        The function slices and maps the tensors from the input dictionary (`std_io_dict`) into a structured format,
        where each row corresponds to a specific inference ID and is repeated according to the number of attention heads (`num_heads`).
        This is useful for models that require multi-head attention or repeated tensor structures.

        The process can be visualized as follows:

        1. Input Dictionary (std_io_dict):
        | pred_box | pred_cls | pred_scr | pred_ious | conf_msk | token_ch | image_id | filename |
        |----------|----------|----------|-----------|----------|----------|----------|----------|
        | [box1, box2, ...] | [cls1, cls2, ...] | [scr1, scr2, ...] | [ious1, ious2, ...] | [msk1, msk2, ...] | [ch1, ch2, ...] | [img1, img2, ...] | [file1, file2, ...] |

        plus it does a. Slicing (based on input_infer_id and total_infer_ids) b. Repeat for num_heads

        Parameters:
            std_io_dict (dict): A dictionary containing the following keys:
                - pred_box: List of predicted bounding boxes.
                - pred_cls: List of predicted classes.
                - pred_scr: List of prediction scores.
                - pred_ious: List of predicted IoU scores.
                - conf_msk: List of confidence masks.
                - token_ch: List of token channels (optional).
                - image_id: List of image IDs.
                - filename: List of filenames.
            input_infer_id (int): The current inference ID to slice the tensors.
            total_infer_ids (int): The total number of inference IDs.
            num_heads (int): The number of attention heads to repeat the tensor structure.

        Returns:
            dict: A dictionary with the following keys, where each value is a list of repeated and sliced tensors:
                - stdio_imageid: Repeated image IDs.
                - stdio_filename: Repeated filenames.
                - stdio_pred_box: Repeated and sliced predicted bounding boxes.
                - stdio_pred_cls: Repeated and sliced predicted classes.
                - stdio_pred_scr: Repeated and sliced prediction scores.
                - stdio_pred_ious: Repeated and sliced predicted IoU scores.
                - stdio_conf_msk: Repeated and sliced confidence masks.
                - token_ch: Repeated and sliced token channels (if available).
        """
        std_io_dict = asdict(std_io)
        batch_size = std_io_dict["pred_box"]
        infer_slicing_step_size = len(std_io_dict["pred_box"][0]) // total_infer_ids
        slice_prev_ptr = input_infer_id * infer_slicing_step_size
        slice_fwd_ptr = (input_infer_id + 1) * infer_slicing_step_size
        ## using pred_box as prediction here but it changes from model to model
        sliced_preds = [
            a[slice_prev_ptr:slice_fwd_ptr] for a in std_io_dict["pred_box"]
        ]

        stdio_imageid = [
            img_id
            for img_id, boxes in zip(std_io_dict["image_id"], sliced_preds)
            for _ in range(num_heads)
            for _ in boxes
        ]

        stdio_filename = [
            filename
            for filename, boxes in zip(std_io_dict["filename"], sliced_preds)
            for _ in range(num_heads)
            for _ in boxes
        ]

        stdio_pred_box = [
            box
            for boxes in std_io_dict["pred_box"]
            for _ in range(num_heads)
            for box in boxes[slice_prev_ptr:slice_fwd_ptr]
        ]
        stdio_pred_cls = [
            cls
            for clss in std_io_dict["pred_cls"]
            for _ in range(num_heads)
            for cls in clss[slice_prev_ptr:slice_fwd_ptr]
        ]
        stdio_pred_scr = [
            scr
            for scrs in std_io_dict["pred_scr"]
            for _ in range(num_heads)
            for scr in scrs[slice_prev_ptr:slice_fwd_ptr]
        ]
        stdio_pred_ious = [
            ious
            for ious_list in std_io_dict["pred_ious"]
            for _ in range(num_heads)
            for ious in ious_list[slice_prev_ptr:slice_fwd_ptr]
        ]
        stdio_conf_msk = [
            msk
            for msks in std_io_dict["conf_msk"]
            for _ in range(num_heads)
            for msk in msks[slice_prev_ptr:slice_fwd_ptr]
        ]
        token_ch = (
            [
                ious
                for ious_list in std_io_dict["token_ch"]
                for _ in range(num_heads)
                for ious in ious_list[slice_prev_ptr:slice_fwd_ptr]
            ]
            if std_io.token_ch is not None
            else [None] * len(stdio_pred_box)
        )

        # Combine into the final dictionary (if needed)
        expand_std_io_dict = {
            "image_id": stdio_imageid,
            "filename": stdio_filename,
            "pred_box": stdio_pred_box,
            "pred_cls": stdio_pred_cls,
            "pred_scr": stdio_pred_scr,
            "pred_ious": stdio_pred_ious,
            "conf_msk": stdio_conf_msk,
            "token_ch": token_ch,
        }
        return expand_std_io_dict

    def _process_det_decoder_resid_subpattern(
        self,
        input_infer_id: int,
        total_infer_ids: int,
        subpattern: str,
        feature: Tensor,
        std_io: IOFrame,
    ) -> Dict[str, list]:
        match subpattern:
            case "pre_img" | "pos_img" | "refpnts":
                # feature = feature.permute(
                #     1, 0, 2
                # )  # get the batch_size to the outer index
                batch_size, num_features, feature_dim = feature.shape
                expand_std_io_dict = self._expand_std_io(
                    input_infer_id, total_infer_ids, std_io
                )
                flatten_feature = feature.reshape(-1, feature_dim)
                token_id = (
                    torch.arange(num_features, dtype=torch.int16)
                    .repeat(batch_size, 1)
                    .reshape(-1)
                )
                heads_id = torch.full(
                    (num_features * batch_size,), -1, dtype=torch.int16
                )
                infer_id = torch.full(
                    (num_features * batch_size,), input_infer_id, dtype=torch.int16
                )

                processed_result = {
                    "infer_id": infer_id.tolist(),
                    "heads_id": heads_id.tolist(),
                    "token_id": token_id.tolist(),
                    "features": flatten_feature.tolist(),
                }

                result = {**expand_std_io_dict, **processed_result}
                return result

            case "pre_txt" | "pos_txt":
                ## ToDo: need to validate this part
                # feature = feature.permute(1,0,2) # get the batch_size to the outer index
                batch_size, num_features, feature_dim = feature.shape
                expand_std_io_dict = self._expand_std_io(
                    input_infer_id, total_infer_ids, std_io
                )
                flatten_feature = feature.reshape(-1, feature_dim)
                token_id = (
                    torch.arange(num_features, dtype=torch.int16)
                    .repeat(batch_size, 1)
                    .reshape(-1)
                )
                heads_id = torch.full(
                    (num_features * batch_size,), -1, dtype=torch.int16
                )
                infer_id = torch.full(
                    (num_features * batch_size,), input_infer_id, dtype=torch.int16
                )

                processed_result = {
                    "infer_id": infer_id.tolist(),
                    "heads_id": heads_id.tolist(),
                    "token_id": token_id.tolist(),
                    "features": flatten_feature.tolist(),
                }

                result = {**expand_std_io_dict, **processed_result}
                return result
            case unsupported:
                raise NotImplementedError(f"unsupported sub_pattern: {unsupported}.")

    def process_det_decoder_resid(
        self, recorder: Recorder, std_io: IOFrame, num_heads: int = 1
    ) -> Dict[Tuple[int, str], List[Dict]]:
        return self._decompose_with(
            self._process_det_decoder_resid_subpattern, recorder, std_io
        )

    def _process_det_decoder_mha_subpattern(
        self,
        input_infer_id: int,
        total_infer_ids: int,
        subpattern: str,
        feature: Tensor,
        std_io: IOFrame,
    ) -> Dict[str, list]:
        match subpattern:
            case "weights":
                batch_size, num_heads, num_features, feature_dim = feature.shape
                flatten_feature = feature.reshape(-1, feature_dim)
                expand_std_io_dict = self._expand_std_io(
                    input_infer_id, total_infer_ids, std_io, num_heads
                )
                token_id = (
                    torch.arange(num_features, dtype=torch.int16)
                    .repeat(num_heads * batch_size, 1)
                    .reshape(-1)
                )
                heads_id = (
                    torch.arange(num_heads, dtype=torch.int16)
                    .repeat(num_features * batch_size, 1)
                    .reshape(-1)
                )
                infer_id = torch.full(
                    (num_features * num_heads * batch_size,),
                    input_infer_id,
                    dtype=torch.int16,
                )

                processed_result = {
                    "infer_id": infer_id.tolist(),
                    "heads_id": heads_id.tolist(),
                    "token_id": token_id.tolist(),
                    "features": flatten_feature.tolist(),
                }

                result = {**expand_std_io_dict, **processed_result}
                return result

            case "outputs":
                feature = feature.permute(1, 0, 2)  # get the batch_size to the outer index
                batch_size, num_features, feature_dim = feature.shape
                expand_std_io_dict = self._expand_std_io(
                    input_infer_id, total_infer_ids, std_io
                )
                flatten_feature = feature.reshape(-1, feature_dim)
                token_id = (
                    torch.arange(num_features, dtype=torch.int16)
                    .repeat(batch_size, 1)
                    .reshape(-1)
                )
                heads_id = torch.full(
                    (num_features * batch_size,), -1, dtype=torch.int16
                )
                infer_id = torch.full(
                    (num_features * batch_size,), input_infer_id, dtype=torch.int16
                )

                processed_result = {
                    "infer_id": infer_id.tolist(),
                    "heads_id": heads_id.tolist(),
                    "token_id": token_id.tolist(),
                    "features": flatten_feature.tolist(),
                }

                result = {**expand_std_io_dict, **processed_result}
                return result

            case unsupported:
                raise NotImplementedError(f"unsupported sub_pattern: {unsupported}.")

    def process_det_decoder_mha(
        self, recorder: Recorder, std_io: IOFrame, num_heads: int = 1
    ) -> Dict[Tuple[int, str], List[Dict]]:
        return self._decompose_with(
            self._process_det_decoder_mha_subpattern, recorder, std_io
        )

    def _process_det_decoder_mlp_subpattern(
        self,
        input_infer_id: int,
        total_infer_ids: int,
        subpattern: str,
        feature: Tensor,
        std_io: IOFrame,
    ) -> Dict[str, list]:
        match subpattern:
            case "pos_img":
                # feature = feature.permute(1, 0, 2)  # get the batch_size to the outer index
                batch_size, num_features, feature_dim = feature.shape
                expand_std_io_dict = self._expand_std_io(
                    input_infer_id, total_infer_ids, std_io
                )
                flatten_feature = feature.reshape(-1, feature_dim)
                token_id = (
                    torch.arange(num_features, dtype=torch.int16)
                    .repeat(batch_size, 1)
                    .reshape(-1)
                )
                heads_id = torch.full(
                    (num_features * batch_size,), -1, dtype=torch.int16
                )
                infer_id = torch.full(
                    (num_features * batch_size,), input_infer_id, dtype=torch.int16
                )

                processed_result = {
                    "infer_id": infer_id.tolist(),
                    "heads_id": heads_id.tolist(),
                    "token_id": token_id.tolist(),
                    "features": flatten_feature.tolist(),
                }

                result = {**expand_std_io_dict, **processed_result}
                return result
            case unsupported:
                raise NotImplementedError(f"unsupported sub_pattern: {unsupported}.")

    def process_det_decoder_mlp(
        self, recorder: Recorder, std_io: IOFrame, num_heads: int = 1
    ) -> Dict[Tuple[int, str], List[Dict]]:
        return self._decompose_with(
            self._process_det_decoder_mlp_subpattern, recorder, std_io
        )

    def process_io(self, recorder: Recorder) -> IOFrame:
        records = recorder.fetch_records()["io"][0]
        assert "batched_inputs" in records, "require batched_inputs."

        bis = records["batched_inputs"]
        sio = IOFrame(
            image_id=[str(bi["image_id"]) for bi in bis],
            filename=[str(bi["file_name"]) for bi in bis],
            # true_box=[bi["instances"].gt_boxes.tensor.detach().cpu().tolist() for bi in bis],
            # true_cls=[bi["instances"].gt_classes.detach().cpu().tolist() for bi in bis],
        )

        if "batched_outputs" in records:
            bos = records["batched_outputs"]
            sio.pred_box = [
                bo["unprocessed_boxes"].tensor.detach().cpu().numpy() for bo in bos
            ]
            sio.pred_cls = [
                bo["unprocessed_clsid"].detach().cpu().numpy() for bo in bos
            ]
            sio.pred_scr = [
                bo["unprocessed_score"].detach().cpu().numpy() for bo in bos
            ]

        if "confusion_mask" in records:
            conf_mask = records["confusion_mask"]
            sio.conf_msk = [conf_msk.detach().cpu().numpy() for conf_msk in conf_mask]
        if "pred_ious" in records:
            pred_ious = records["pred_ious"]
            sio.pred_ious = [
                _pred_ious.detach().cpu().numpy() for _pred_ious in pred_ious
            ]
        return sio
