# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
import time
import json
from typing import Dict, Any
from PIL import Image
import torch
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from blueglass.configs import BLUEGLASSConf

from google import genai
from google.genai import types, errors

from .base import ClosedSourceModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKOFF_TIME_IN_SEC = 5
EXTENDED_WAIT_TIME_IN_SEC = 5 * 60

logger = setup_blueglass_logger(__name__)


class Gemini(ClosedSourceModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        assert conf.model.api_key is not None, "missing api_key for Gemini."
        self.api = genai.Client(api_key=conf.model.api_key)
        self.model_name = "gemini-2.0-flash-exp"
        self.prompt = "Detect 2d bounding boxes."
        instruct = (
            "Return bounding boxes as a JSON array with labels. Never return masks or code fencing."
            "Note there could be multiple instances of a object. Important! - Only output 100 or less objects,"
            "the json array should not contain more than 100 dictionaries or objects."
        )
        safety_opts = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            )
        ]
        self.opts = types.GenerateContentConfig(
            system_instruction=instruct,
            temperature=0.5,
            safety_settings=safety_opts,
        )

    def send_request(self, frame: Image.Image, prompt: str) -> Dict[str, Any]:
        try:
            resp = self.api.models.generate_content(
                model=self.model_name, contents=[frame, prompt], config=self.opts
            )
        except errors.ServerError as e:
            print(f"received server error, switched to extended wait.\n{e}")
            time.sleep(EXTENDED_WAIT_TIME_IN_SEC)
            resp = self.api.models.generate_content(
                model=self.model_name, contents=[frame, prompt], config=self.opts
            )
        except Exception as e:
            logger.info("unexpected error, skipped sample.")
            return ""

        return resp.text

    def _sanitized(self, resp: str):
        """
        Removes extra text and code fences.
        Handle when the output is terminated.
        """
        if "json" not in resp:
            resp = "```json\n[\n]\n```"

        lines = resp.splitlines()

        if resp.startswith("```") and not resp.endswith("```"):
            # remove the last 5 entries.
            lines = lines[:-5]

            # remove the trailing comma.
            lines[-1] = lines[-1][:-1]

            # append with correct end.
            lines.extend(["]", "```"])

        for i, line in enumerate(lines):
            if line == "```json":
                # remove start fence.
                resp = "\n".join(lines[i + 1 :])

                # remove end fence.
                resp = resp.split("```")[0]
                break

        return resp

    def postprocess(self, inputs: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Converts the response text to d2 format.

        Expected response: "```json\n [{"box_2d": [...], "label": str}, ...\n```]
        """
        im_h, im_w = inputs["height"], inputs["width"]

        try:
            preds = json.loads(self._sanitized(response))
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

        boxes = torch.tensor(boxes, dtype=torch.long, device=DEVICE)

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
