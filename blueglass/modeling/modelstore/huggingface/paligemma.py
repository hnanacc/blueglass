# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import json
from typing import List, Dict, Any
import torch
from transformers.image_utils import load_image
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from blueglass.configs import BLUEGLASSConf
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from .base import HFModel


TOKEN_LIMIT = 1024


class PaliGemma2(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.hf_id = "google/paligemma2-3b-pt-448"
        self.procr = PaliGemmaProcessor.from_pretrained(self.hf_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.hf_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

        self.instruction_prompt = "<image> Detect 2D bounding boxes for all objects in the image. Output them in json format."

    @torch.inference_mode()
    def forward(self, batched_inputs: List[Dict[str, Any]]):
        assert len(batched_inputs) == 1, "HF models only support batch size 1."
        inputs = batched_inputs[0]
        del inputs["image"]

        im = load_image(inputs["file_name"])

        proc_inputs = self.procr(
            text=self.instruction_prompt, images=im, return_tensors="pt"
        ).to(device=self.device, dtype=torch.bfloat16)

        batched_tokens = self.model.generate(
            **proc_inputs, max_new_tokens=TOKEN_LIMIT, do_sample=False
        )

        next_tokens = batched_tokens[0][proc_inputs["input_ids"].shape[-1] :]
        response = self.procr.decode(next_tokens, skip_special_tokens=True)

        return self.postprocess(inputs, response)

    def postprocess(self, inputs: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Converts the response text to d2 format.

        Expected response: "```json\n [{"box_2d": [...], "label": str}, ...\n```]
        """
        im_h, im_w = inputs["height"], inputs["width"]

        try:
            preds = json.loads(response)
        except json.JSONDecodeError as e:

            print(f"invalid json:\n {response}\nproduced error: {e}")
            preds = []
        except Exception as e:
            print(f"unexpected error:\n{e}")
            preds = []

        boxes, clses = [], []
        for pr in preds:
            y1, x1, y2, x2 = pr["box_2d"]
            abs_y1 = int(y1 / 1000 * im_h)
            abs_x1 = int(x1 / 1000 * im_w)
            abs_y2 = int(y2 / 1000 * im_h)
            abs_x2 = int(x2 / 1000 * im_w)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1

            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
            clses.append(pr["label"].strip())

        boxes = torch.tensor(boxes, dtype=torch.long, device=self.device)

        if len(boxes) > 0:
            assert torch.all(boxes[:, 0] <= boxes[:, 2]), "invalid x coords."
            assert torch.all(boxes[:, 1] <= boxes[:, 3]), "invalid y coords."

        return {
            "instances": Instances(
                (im_h, im_w),
                pred_boxes=Boxes(boxes),
                pred_object_descriptions=Descriptions(clses),
                pred_classes=torch.zeros(len(boxes)),
                scores=torch.ones(len(boxes)),
            )
        }
