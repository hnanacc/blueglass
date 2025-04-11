# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import List, Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.image_utils import load_image
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from blueglass.configs import BLUEGLASSConf

from .base import HFModel


TOKEN_LIMIT = 1024


class Florence(HFModel):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.args = conf
        self.hf_id = "microsoft/Florence-2-large-ft"
        self.procr = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.hf_id, torch_dtype=torch.float32, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        self.task = "<OD>"
        self.prompt = self.task

    def preprocess(self, inputs: Dict[str, Any]):
        del inputs["image"]
        return load_image(inputs["file_name"])

    @torch.inference_mode()
    def forward(self, batched_inputs: List[Dict[str, Any]]):
        assert len(batched_inputs) == 1, "HF models only support batch size 1."
        inputs = batched_inputs[0]
        im = self.preprocess(inputs)
        proc_input = self.procr(text=self.prompt, images=im, return_tensors="pt").to(
            device=self.device
        )

        batched_outputs = self.model.generate(
            **proc_input,
            max_new_tokens=TOKEN_LIMIT,
            do_sample=False,
            num_beams=3,
        )

        response = self.procr.batch_decode(
            batched_outputs.sequences, skip_special_tokens=False
        )[0]
        response = self.procr.post_process_generation(
            response, task=self.task, image_size=(inputs["width"], inputs["height"])
        )

        return [self.postprocess(inputs, response[self.task])]

    def register_features(self, hidden_states):
        pass

    def postprocess(self, inputs: Dict[str, Any], response: Dict[str, Any]):
        im_h, im_w = inputs["height"], inputs["width"]

        boxes = Boxes(torch.tensor(response["bboxes"], device=self.device))
        clses = Descriptions(response["labels"])

        return {
            "instances": Instances(
                (im_h, im_w),
                pred_boxes=boxes,
                pred_object_descriptions=clses,
                scores=torch.ones(len(boxes), dtype=torch.float, device=self.device),
            )
        }
