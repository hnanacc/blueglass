# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import copy
from PIL import Image
import torch
from torch import nn
from blueglass.third_party.detectron2.structures import Instances, Boxes
from blueglass.structures.descriptions import Descriptions
from typing import List, Dict


def add_llava_arguments(subparser):
    p = subparser.add_parser("llava", help="Executes LLaVA-OneVision.")
    p.add_argument(
        "--model-checkpoint-path", type=str, help="HF or local path of model."
    )
    p.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )
    p.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="No. of beams to sample during inference.",
    )
    p.add_argument(
        "--top-p", type=float, default=0.7, help="Top_P during inference sampling."
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens to generate.",
    )
    p.add_argument(
        "--prompt-mode",
        type=str,
        default="basic",
        choices=["basic", "fewshot"],
        help="Mode of prompts to the model.",
    )
    p.add_argument(
        "--fewshot-freq",
        type=int,
        default=5,
        help="No. of few shot examples to use per query.",
    )


INSTRUCTION = "{}\n\
Locate all objects in the image and give their bounding box and classname\
For each object provide the answer in the format xleft,yleft,xright,yright:classname\
with one object on each line. xleft, yleft are the top-left coordinates of the box\
and xright,yright are the bottom-right coordinates of the box. Do not group objects."


class LLaVA(nn.Module):
    def __init__(self, args):
        super().__init__()

        from blueglass.modeling.benchmarks.llava.llava.model.builder import (
            load_pretrained_model,
        )
        from blueglass.modeling.benchmarks.llava.llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )

        self.args = args
        self.device = torch.device(
            "cuda:0" if torch.cuda.device_count() > 1 else "cuda"
        )

        self.tokenizer, self.model, self.im_preprocessor, self.ctx_size = (
            load_pretrained_model(
                args.model_checkpoint_path,
                None,
                "llava_qwen",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        )
        self.model.eval()

        self.im_token_str = DEFAULT_IMAGE_TOKEN
        self.im_token_idx = IMAGE_TOKEN_INDEX

        self.instr = self.prepare_conversation()

    def prepare_conversation(self):
        from blueglass.modeling.benchmarks.llava.llava.conversation import (
            conv_templates,
        )

        conv = copy.deepcopy(conv_templates["qwen_2"])
        instruction = INSTRUCTION.format(self.im_token_str)
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.inference_mode()
    def forward(self, batched_inputs: List[Dict]):
        batched_outputs = []
        for bi in batched_inputs:
            input_ids, images, image_sizes = self.preprocess(bi)

            output_ids = self.model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_new_tokens,
            )
            outputs = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            batched_outputs.append(outputs[0])

        return self.postprocess(batched_inputs, batched_outputs)

    def preprocess(self, bi: Dict):
        from blueglass.modeling.benchmarks.llava.llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )

        bi["image"] = Image.open(bi["file_name"])

        images = [bi["image"]]
        image_sizes = [im.size for im in images]

        images = process_images(images, self.im_preprocessor, self.model.config)
        images = [im.half().to(self.device) for im in images]

        return (
            tokenizer_image_token(
                self.instr,
                self.tokenizer,
                self.im_token_idx,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.device),
            images,
            image_sizes,
        )

    def parse_bboxes_and_cnames(self, output: str):
        lines = output.strip().split("\n")
        bboxes, cnames = [], []

        for line in lines:
            bb, cn = line.strip().split(":")
            bb = [float(c.strip()) for c in bb.split(",")]
            cn = cn.strip().lower()
            bboxes.append(bb)
            cnames.append(cn)

        return Boxes(bboxes), cnames

    def postprocess(self, batched_inputs: List[Dict], batched_outputs: List[str]):
        instances = []

        for bi, bo in zip(batched_inputs, batched_outputs):
            bboxes, labels = self.parse_bboxes_and_cnames(bo)
            bboxes.scale(bi["width"], bi["height"])

            inst = Instances((bi["height"], bi["width"]))
            inst.pred_boxes = bboxes.to(self.device)
            inst.pred_object_descriptions = Descriptions(labels)

            inst.scores = [1.0]

            instances.append({"instances": inst})

        return instances
