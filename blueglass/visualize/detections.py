# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import cv2
import numpy as np
from torch import Tensor
from typing import List, Union, Tuple
from blueglass.third_party.detectron2.utils.visualizer import Visualizer, VisImage
from blueglass.third_party.detectron2.structures import Boxes


def _convert_image_or_path(image):
    if isinstance(image, Tensor):
        assert image.shape[0] == 3, "Invalid shape for torch.Tensor image."
        return image.permute(1, 2, 0).cpu().detach().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "Invalid shape for ndarray image."
        return image
    if isinstance(image, str):
        assert os.path.exists(image), "Image path doesn't exits."
        return cv2.imread(image)[:, :, ::-1]


def visualize_truth(path: str, image: Tensor, bboxes: Tensor, cnames: List[str]):
    im = _convert_image_or_path(image)
    Visualizer(im).overlay_instances(boxes=bboxes, labels=cnames).save(path)


def visualize_detections(
    path: str,
    image: Union[np.ndarray, Tensor, str],
    bboxes: Union[Boxes, Tensor, List[Tuple[float, float, float, float]]],
    cnames: List[str],
    scores: List[float],
):
    im = _convert_image_or_path(image)
    lb = [f"{cn.lower()} {sc * 100:.1f}%" for cn, sc in zip(cnames, scores)]
    Visualizer(im).overlay_instances(boxes=bboxes, labels=lb).save(path)


def visualize_detections_with_truth(
    path: str,
    image: Union[np.ndarray, Tensor, str],
    dt_bboxes: Tensor,
    dt_cnames: List[str],
    scores: Tensor,
    gt_bboxes: Tensor,
    gt_cnames: List[str],
):
    dt_mark = [f"{cn.lower()} {sc * 100:.1f}%" for cn, sc in zip(dt_cnames, scores)]
    gt_mark = [cn.lower() for cn in gt_cnames]

    im = _convert_image_or_path(image)
    dt = Visualizer(im).overlay_instances(boxes=dt_bboxes, labels=dt_mark).get_image()
    gt = Visualizer(im).overlay_instances(boxes=gt_bboxes, labels=gt_mark).get_image()

    VisImage(np.concatenate([dt, gt], axis=0)).save(path)


def draw_embed_sim(row_embeds: Tensor, col_embeds: Tensor):
    return
