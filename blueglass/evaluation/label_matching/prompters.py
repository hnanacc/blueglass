# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from abc import ABC, abstractmethod
import copy
from typing import List
from blueglass.configs import Prompter as PrompterVariant


class Prompter(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def to_prompts(self, descs: List[str]) -> List[str]:
        pass


class BasicPrompter(Prompter):
    def to_prompts(self, descs: List[str]):
        return [f"a {d.replace('_', ' ').lower()}" for d in descs]


class LearnedPrompter(Prompter):
    def __init__(self, num_tokens: int = 4, distribute: bool = False, **kwargs):
        self.num_tokens = num_tokens
        self.distribute = distribute

    def to_prompts(self, descs: List[str]):
        pass


class BootstrapPrompter(Prompter):
    def __init__(self, **kwargs):
        pass

    def to_prompts(self, descs: List[str]):
        pass


class BasicVLMPrompter(Prompter):
    def to_prompts(self) -> List[str]:
        return ["Locate all objects in the given image."]


class FewShotPrompter:
    def __init__(self):
        self.conv = conv

        self.im_token_rep = im_token_rep
        self.im_token_idx = im_token_idx

        self.queries, self.responses = self.prepare_examples()
        self.instruction = "List all the objects in the image along with their classes."

    def prepare_examples(self, bboxes, classnames):
        queries, responses = [], []
        return queries, responses

    def format_instances(self):
        return

    def to_prompt(self, batched_bboxes: List, batched_classnames: List[str]):
        conv = copy.deepcopy(self.conv_template)

        for bb, cn in zip(batched_bboxes, batched_classnames):
            conv.append_message(
                conv.roles[0], f"{self.im_token_def}\n{self.instruction}"
            )
            conv.append_message(conv.roles[1], self.format_instances(bb, cn))

        conv.append_message(conv.roles[0], f"{self.im_token_def}\n{self.instruction}")
        conv.append_message(conv.roles[1], None)

        return conv.get_prompt()


def prepare_prompter(variant: PrompterVariant, **kwargs) -> Prompter:
    match variant:
        case PrompterVariant.BASIC:
            return BasicPrompter(**kwargs)
        case unsupported:
            raise ValueError(f"unsupported prompter: {unsupported}")
