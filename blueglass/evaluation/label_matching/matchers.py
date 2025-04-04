# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from copy import deepcopy
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from .encoders import prepare_text_encoder
from .prompters import prepare_prompter
from blueglass.configs import (
    Matcher as MatcherVariant,
    Prompter as PrompterVariant,
    Encoder as EncoderVariant,
)


class Matcher(ABC):
    """
    Matcher
    ---

    Expects
    ---
    pos_cnames: class names of positive classes.
    neg_cnames: class names of negative classes.

    Returns
    ---
    class_ids: IDs of classes [0, N) U {-1}. -1 denotes negative class.
    prob_conf: confidence assigned to each similarity match.

    """

    @abstractmethod
    def to_cids_and_scores(self, descs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class PromptMatcher(Matcher):
    PARTS_TMPL = "part of {}"
    NEG_CNAMES = ["object", "thing", "background in an image"]

    def __init__(
        self,
        pos_cnames: List[str],
        prompter_name: PrompterVariant,
        encoder_name: EncoderVariant,
        use_negatives: bool,
        use_parts: bool,
        num_topk_matches: int,
    ):
        self.encoder = prepare_text_encoder(encoder_name)
        self.prompter = prepare_prompter(prompter_name)
        self.num_topk_matches = num_topk_matches
        self.use_negatives = use_negatives
        self.use_parts = use_parts
        self.pos_end_index = len(pos_cnames) - 1

        self.cmb_cnames = deepcopy(pos_cnames)

        if self.use_negatives:
            self.cmb_cnames += PromptMatcher.NEG_CNAMES

        if self.use_parts:
            self.cmb_cnames += [
                PromptMatcher.PARTS_TMPL.format(c) for c in self.cmb_cnames
            ]

        self.cmb_prompts = self.prompter.to_prompts(self.cmb_cnames)

    def to_cids_and_scores(self, descs: List[str]):
        desc_prompts = self.prompter.to_prompts(descs)
        similarities = self.encoder.compute_similarities(
            desc_prompts, self.cmb_prompts
        ).topk(min(self.num_topk_matches, len(self.cmb_prompts)))
        similarities.indices[similarities.indices > self.pos_end_index] = -1
        return similarities.indices, similarities.values


class LLMMatcher(Matcher):
    def __init__(
        self,
        pos_cnames: List[str],
        prompter_name: PrompterVariant,
        encoder_name: EncoderVariant,
        use_parts: bool = False,
        use_negatives: bool = False,
        **kwargs,
    ):
        self.pos_cnames = pos_cnames
        self.llm_client = self.build_llm_client(encoder_name)

    def build_llm_client(self, encoder_name: str):
        return encoder_name

    def to_cids_and_scores(self, descs: List[str]):
        return [], []


def prepare_matcher(
    variant: MatcherVariant,
    classnames: List[str],
    prompter_name: PrompterVariant,
    encoder_name: EncoderVariant,
    use_negatives: bool,
    use_parts: bool,
    num_topk_matches: int,
    **kwargs,
) -> Matcher:
    match variant:
        case MatcherVariant.LLM:
            return LLMMatcher(classnames, prompter_name, encoder_name, **kwargs)
        case MatcherVariant.EMBED_SIM:
            return PromptMatcher(
                classnames,
                prompter_name,
                encoder_name,
                use_negatives,
                use_parts,
                num_topk_matches,
                **kwargs,
            )
        case unsupported:
            raise ValueError(f"unsupported matcher: {unsupported}.")
