# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import cv2
from blueglass.utils.logger_utils import setup_blueglass_logger
import base64
from io import BytesIO
import json
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
import torch
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from blueglass.configs import BLUEGLASSConf
from blueglass.visualize.detections import visualize_detections
from .base import ClosedSourceModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKOFF_TIME_IN_SEC = 5
EXTENDED_WAIT_TIME_IN_SEC = 5 * 60

OAI_OUTPUT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "detection_outputs",
        "schema": {
            "type": "object",
            "properties": {
                "preds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box2d": {"type": "array", "items": {"type": "number"}},
                            "label": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

logger = setup_blueglass_logger(__name__)


class GPT4oMini(ClosedSourceModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        assert conf.model.api_key is not None, "missing api_key for GPT 4o mini."
        self.api = OpenAI(api_key=conf.model.api_key)
        self.model_name = "gpt-4o-mini"
        self.instruct = (
            "Return bounding boxes as a JSON array with labels. Never return masks or code fencing."
            "Note there could be multiple instances of a object. Important! - Only output 100 or less objects,"
            "the json array should not contain more than 100 dictionaries or objects."
            "The bounding boxes should in the format [x_top_left, y_top_left, x_bottom_right, y_bottom_right]"
            "and in normalized format where if the image size is 1000 and the absolute x_top_left coordinate"
            "for an object is 500, then the output x_top_left should be 500 / 1000 = 0.5 and it falls between 0 to 1."
        )
        self.prompt = "Detect 2d bounding boxes"

        logger.info(
            """NOTE: OpenAI models still do not support object grounding.
        We still prompt them to produce images and save a sample and
        return predictions. So, the evaluation is inappropiate. 
        """
        )

    def prepare_chat(self, frame: Image.Image, prompt: str):
        buffer = BytesIO()
        frame.save(buffer, format="jpeg")
        b64_frame = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return [
            {"role": "system", "content": self.instruct},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"},
                    }
                ],
            },
        ]

    def send_request(self, frame: Image.Image, prompt: str):
        resp = self.api.chat.completions.create(
            model=self.model_name,
            messages=self.prepare_chat(frame, prompt),
            response_format=OAI_OUTPUT_FORMAT,
        )
        return resp.choices[0].message.content

    def postprocess(self, inputs: Dict[str, Any], response: str) -> Dict[str, Any]:
        """NOTE: OpenAI models still do not support object grounding.
        We still prompt them to produce images and save a sample and
        return empty predictions. So, the evaluation is just visualization.
        """
        im_h, im_w = inputs["height"], inputs["width"]

        preds = json.loads(response)["preds"]
        boxes = [p["box2d"] for p in preds]
        clses = [p["label"] for p in preds]

        boxes = [(b[0] * im_w, b[1] * im_h, b[2] * im_w, b[3] * im_h) for b in boxes]

        fn = inputs["file_name"]
        im = cv2.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        visualize_detections(os.path.basename(fn), im, boxes, clses, [1.0] * len(boxes))

        return {
            "instances": Instances(
                (im_h, im_w),
                pred_boxes=Boxes(torch.tensor(boxes, device=self.device)),
                pred_object_descriptions=Descriptions(clses),
                pred_classes=torch.zeros(len(boxes), device=self.device),
                scores=torch.ones(len(boxes), device=self.device),
            )
        }
