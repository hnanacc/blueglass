# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from blueglass.modeling.modelstore.grounding_dino.groundingdino.util import (
    get_tokenlizer,
)
from blueglass.modeling.modelstore.grounding_dino.groundingdino.util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed

from blueglass.features import intercept_manager
from blueglass.configs.defaults import BLUEGLASSConf


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        conf: BLUEGLASSConf = None,
        device=None,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.conf = conf

        # add device attribute, keep in mind the gotchas.
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(
            self.bert.config.hidden_size, self.hidden_dim, bias=True
        )
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [
            _class_embed for i in range(transformer.num_decoder_layers)
        ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, "features"):
            del self.features
        if hasattr(self, "poss"):
            del self.poss

    def set_image_features(self, features, poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, targets: List | None = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        self.blueglass_conf = kw["blueglass_conf"]
        # ---------------------------
        # TEXT ENCODING USING BERT.
        # ---------------------------

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            self.device
        )

        # self_attention_mask shape
        # (bs, token_seq_len, token_seq_len)
        # (for rules see annotations inside the function.)

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        # limit the max token sequence length.
        # TODO: why limit the text length?
        # this is a problem because the for long-tail
        # datasets we have too many classes with token
        # sequence lenght reaching 4800.

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : self.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : self.max_text_len
            ]

        # extract text embeddings
        if self.sub_sentence_present:
            # in our case, we *have* sub-sentences,
            # essentially, each class name is a sub-sentence.
            # so we use a different attention mask and position_ids,
            # than generated by the standard hf tokenizer.
            # i.e. self attention masks generated on line 290.
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        # we encode the caption using bert and use
        # the last_hidden_state (bs, token_seq_len, bert_d_model)
        # as the representation embedding for the whole caption.
        # where bert_d_model is 768.
        # -> so for each token, we get a embedding vector.

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        # the d_model of bert is 768, hence,
        # the output embeddings are of shape 768.
        # whereas the d_model of detr is 256.
        # so we need to map the text features from 768
        # to 256 size and we do that using a linear layer.

        encoded_text = self.feat_map(
            bert_output["last_hidden_state"]
        )  # bs, 195, d_model

        # this mask is from hf tokenizer (line 282)
        # this represents the token which are for text vs padding.
        # 1 for text and 0 for added padding to make the sequences
        # of same length (longest padding).

        text_token_mask = tokenized.attention_mask.bool()  # bs, 195

        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        # we already limit things to max_text_len
        # so encoded text cannot be different. But still
        # kinda ensures it. not necessary!
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        # definitions.
        # encoded_text -> feature output from last bert layer.
        # text_token_maks -> padding vs text tokens mask.
        # position_ids -> position ids with sub-sentence in mind.
        # text_self_attention_masks -> self attention maps.

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195, 195
        }

        # ---------------------------
        # IMAGE ENCODING USING SWIN.
        # ---------------------------

        # import ipdb; ipdb.set_trace()

        # convert list[images(tensors)]
        # to NestedTensor for batched processing.
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # extract features from self.backbone
        # and set them to self.features and self.poss.
        if not hasattr(self, "features") or not hasattr(self, "poss"):
            self.set_image_tensor(samples)

        # the output of swin backbone is a list
        # of features (last 3 of 4 total in this case)
        # which is a NestedTensor. We decompose each feature
        # into source and corresponding mask and then
        # project it to required dim of 256 using Conv2d
        # layer followed by group normalization.

        srcs, masks = [], []
        for l, feat in enumerate(self.features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # num output features = 3 (controlled by config.)
        # but the num_feature_levels = 4.
        # this is useful when we want to extend the backbone and
        # add more feature layers. for the first extension,
        # we use the feature from the last layer of the backbone
        # and pass it through our custom project layers.
        # in this case, it's a conv2d followed by group norm.
        # for rest of the added layers we use the features from
        # the previous layers and do the same (projections).
        # position encoding is sine.

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](self.features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        # ---------------------------
        # PROPOSAL GENERATION (DETR).
        # ---------------------------

        # proposals from the detr transformer.

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs,
            masks,
            input_query_bbox,
            self.poss,
            input_query_label,
            attn_mask,
            text_dict,
            blueglass_conf=self.blueglass_conf,
        )

        # here we get layer wise features and reference_points.
        # this should be equal to the references from the last layer of
        # the deocder because we are using the same box_embed and the same
        # process.

        # deformable-detr-like anchor update

        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)

            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)

        outputs_coord_list = torch.stack(outputs_coord_list)

        # this is the constrastive embed, same as before.
        # returns the similarity scores for each proposal and all text features.
        outputs_class = []
        for layer_id, (layer_cls_embed, layer_hs) in enumerate(
            zip(self.class_embed, hs)
        ):
            cls_out = layer_cls_embed(layer_hs, text_dict)
            outputs_class.append(cls_out)
        outputs_class = torch.stack(outputs_class)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # just unsets self.features and self.poss.
        # why? this is a flaw because of how features and poss
        # are set. Why set them as attributes? should be a variable.
        # if this is some hack, to use the features later, it's bad
        # could instead return them with the final output.

        unset_image_tensor = kw.get("unset_image_tensor", True)
        if unset_image_tensor:
            self.unset_image_tensor()  ## If necessary

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.register_with_name(module_name="groundingdino")
def build_groundingdino(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        device=args.device,
    )

    return model
