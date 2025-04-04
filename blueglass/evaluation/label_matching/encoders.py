# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    BertModel,
    BertTokenizer,
    AutoTokenizer,
)
from typing import List, Tuple
from blueglass.configs import Encoder as EncoderVariant


class Encoder(ABC):
    def __init__(self):
        self.device = (
            torch.device("cuda")
            if torch.cuda.device_count() < 2
            else torch.device("cuda:1")
        )
        self.batch_size = 1024

    def batched_encode_text(self, seq: List[str]) -> Tensor:
        embeds = []
        for st_idx in range(0, len(seq), self.batch_size):
            en_idx = min(len(seq), st_idx + self.batch_size)
            embedi = self.encode_text(seq[st_idx:en_idx])
            embeds.append(embedi)
        return torch.cat(embeds)

    @abstractmethod
    def encode_text(self) -> Tensor:
        raise NotImplementedError("Please extend this.")

    @abstractmethod
    @torch.inference_mode()
    def compute_similarities(self, queries: List[str], keys: List[str]) -> Tensor:
        raise NotImplementedError("Please extend this.")


class BERT(Encoder):
    def __init__(self, hf_id: str = "bert-base-uncased"):
        super().__init__()
        self.model = BertModel.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, output_hidden_states=True
        ).to(self.device)
        self.procr = BertTokenizer.from_pretrained(hf_id)

    def encode_text(self, seq: List[str], mode="mean") -> Tensor:
        inputs = self.procr(seq, padding=True, return_tensors="pt")
        output = self.model(**inputs.to(self.device))

        if mode == "cls":
            return output.last_hidden_state[:, 0, :]
        if mode == "mean":
            return output.last_hidden_state.mean(dim=1)
        if mode == "max":
            return output.last_hidden_state.max(dim=1).values
        if mode == "pooler":
            return output.pooler_output

        raise ValueError(f"unsupported encode mode: {mode}.")

    def compute_similarities(self, queries: List[str], keys: List[str]) -> Tensor:
        embeds_q = self.batched_encode_text(queries)
        embeds_k = self.batched_encode_text(keys)

        assert embeds_q.shape == (len(queries), 768), "unexpected shape for embeds."
        assert embeds_k.shape == (len(keys), 768), "unexpected shape for embeds."

        sims = (embeds_q @ embeds_k.T) * 100.0

        assert sims.shape == (len(queries), len(keys)), "unexpected shape for sims."

        return sims.softmax(dim=-1)


class NVEmbed(Encoder):
    NVE_INSTRUCTION = "Represent the sentence to retrieve similar sentence: "

    def __init__(self, hf_id: str = "nvidia/NV-Embed-v1"):
        super().__init__()
        assert torch.cuda.device_count() > 1, "need larger or more gpus to fit this."

        self.batch_size = 16
        self.max_token_length = 4096
        self.model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.instruct = NVEmbed.NVE_INSTRUCTION

    def encode_text(self, seq: List[str]):
        assert self.model.device == self.device, "model moved to different device."
        return self.model.encode(
            seq, max_length=self.max_token_length, instruction=self.instruct
        )

    @torch.inference_mode()
    def compute_similarities(self, queries: List[str], keys: List[str]) -> Tensor:
        embeds_q = self.batched_encode_text(queries)
        embeds_k = self.batched_encode_text(keys)

        sims = (embeds_q @ embeds_k.T) * 100.0

        return sims.softmax(dim=-1)


class CLIP(Encoder):
    def __init__(self, hf_id: str = "openai/clip-vit-large-patch14-336"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(hf_id).to(self.device)
        self.procr = CLIPProcessor.from_pretrained(hf_id)

    def encode_text(self, seq: List[str]):
        tokens = self.procr(padding=True, return_tensors="pt", text=seq)
        return self.model.get_text_features(**tokens.to(self.device))

    @torch.inference_mode()
    def compute_similarities(self, queries: List[str], keys: List[str]) -> Tensor:
        embeds_q = self.batched_encode_text(queries)
        embeds_k = self.batched_encode_text(keys)

        sims = (embeds_q @ embeds_k.T) * 100.0

        return sims.softmax(dim=-1)


class SigLIP(Encoder):
    def __init__(self, hf_id: str = "google/siglip-so400m-patch14-384"):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to(
            self.device
        )
        self.procr = AutoProcessor.from_pretrained(hf_id)

        self.temp = 11.82
        self.bias = -12.7

    def encode_text(self, seq: List[str]):
        tokens = self.procr(seq, padding="max_length", return_tensors="pt")
        return self.model.get_text_features(**tokens.to("cuda"))

    @torch.inference_mode()
    def compute_similarities(self, queries: List[str], keys: List[str]):
        embeds_q = self.batched_encode_text(queries)
        embeds_k = self.batched_encode_text(keys)

        embeds_q /= embeds_q.norm(dim=-1, keepdim=True)
        embeds_k /= embeds_k.norm(dim=-1, keepdim=True)

        sims = (embeds_q @ embeds_k.T) * self.temp + self.bias

        return sims.sigmoid()


class B1ade(Encoder):
    def __init__(
        self,
    ):
        super().__init__()
        self.hf_id = "w601sxs/b1ade-embed"
        self.model = AutoModel.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, output_hidden_states=True
        ).to(self.device)
        self.procr = AutoTokenizer.from_pretrained(self.hf_id)

    def encode_text(self, seq: List[str], mode="mean"):
        tokens = self.procr(seq, padding=True, return_tensors="pt")

        output = self.model(**tokens.to(self.device))

        if mode == "cls":
            return output.last_hidden_state[:, 0, :]
        if mode == "mean":
            return output.last_hidden_state.mean(dim=1)
        if mode == "max":
            return output.last_hidden_state.max(dim=1).values
        if mode == "pooler":
            return output.pooler_output

        raise ValueError(f"unsupported encode mode: {mode}.")

    def compute_similarities(self, queries: List[str], keys: List[str]) -> Tensor:
        embeds_q = self.batched_encode_text(queries)
        embeds_k = self.batched_encode_text(keys)

        sims = (embeds_q @ embeds_k.T) * 100.0

        return sims.softmax(dim=-1)


def prepare_text_encoder(variant: EncoderVariant, **kwargs) -> Encoder:
    match variant:
        case EncoderVariant.BERT:
            return BERT(**kwargs)
        case EncoderVariant.B1ADE:
            return B1ade(**kwargs)
        case EncoderVariant.NVEMBED:
            return NVEmbed(**kwargs)
        case EncoderVariant.CLIP:
            return CLIP(**kwargs)
        case EncoderVariant.SIGLIP:
            return SigLIP(**kwargs)

    raise ValueError(f"unsupported encoder variant: {name}")
