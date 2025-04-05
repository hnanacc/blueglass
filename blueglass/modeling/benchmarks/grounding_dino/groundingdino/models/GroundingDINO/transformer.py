# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0
# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional, Dict, Any

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from typing import Optional, Tuple

from blueglass.modeling.benchmarks.grounding_dino.groundingdino.util.misc import (
    inverse_sigmoid,
)

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)

"""
ADD INFORMATION ABOUT THE BLUEGLASS FEATURE USAGE HERE
"""
from blueglass.features import intercept_manager
from blueglass.configs import FeaturePattern, FeatureSubPattern
from blueglass.configs.defaults import BLUEGLASSConf
import numpy as np


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(
                    torch.Tensor(num_feature_levels, d_model)
                )
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        # these are set in the GroundingDINO __init__
        # very bad practise!!!
        # basically class_embed = contrastive_embed
        # bbox_embed = 3 layer mlp with (..., 4) as output.
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(
        self,
        srcs,
        masks,
        refpoint_embed,
        pos_embeds,
        tgt,
        attn_mask=None,
        text_dict=None,
        blueglass_conf: BLUEGLASSConf = None,
    ):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # we further process the features we get from the
        # swin backbone which are of shapes (bs, c, h, w) using a
        # detection transformer architecture, which expects inputs
        # as sequences of shapes (bs, token_seq_len, per_token_size).
        # To convert the features into sequences, we consider every
        # pixel in the feature as a token and because the value of
        # each pixel is described by all values along the channel dim
        # at that pixel location we consider that as the token values.
        # hence, we flatten (bs, c, h, w) -> (bs, h * w, c), converting
        # from image dimensions to sequence dimensions.

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # flatten the hw dimension.
            # (bs, c, h, w) -> (bs, h*w, c)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c

            # similar to position embedding, the transformer doesn't
            # know which level the features are from. Is this a problem? It could
            # be beneficial to also include that information during query
            # processing in the encoder. Here we use an addition "level embedding",
            # along with the position embedding to indicate this information.
            # These are parameters learned during training and are specific to
            # each level. This seems to be the only option. We don't have other
            # heuristic like sine encoding to encode this information, so it is
            # best to learn it during training. Not sure how much this helps,
            # but probably should be beneficial.

            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )

        # because we concatenate all the features from all the levels
        # we need to know where the features from a particular level start.
        # this keeps track of it and is of shape (num_levels,), 4 in our case.
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # we check if there is any feature with empty masks.
        # there should be any but need to see where we use it.
        # TODO.
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################

        # memory is image features or object queries.
        # and memory_text is text features.
        assert text_dict is not None, "unexpected."

        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )

        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        text_dict["encoded_text"] = memory_text

        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()

        # proposal creation and query selection phase.

        if self.two_stage_type == "standard":
            # does 2 things,
            # 1. creates anchors boxes (static proposals) for each pixel.
            # 2. masks invalid proposals and query features.

            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            # this layer learns the anchor selection criteria with a linear
            # layer (identity -> d_model)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            assert (
                self.enc_out_class_embed is not None
            ), "enc_out_class_embed not initialized."
            assert (
                self.enc_out_bbox_embed is not None
            ), "enc_out_bbox_embed not initialized."

            # use the contrastive sim between object queries and the
            # text features and use that to filter the object queries.
            # this is called language guided query selection (in paper).
            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(
                    output_memory, text_dict
                )
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            # we get the max similarity score for each anchor query.

            # get topk
            topk_logits = enc_outputs_class_unselected.max(-1)[0]

            # similarly we get bboxes deltas (..., N) from anchor features.
            # we then add them to the proposals. note, deltas are unsigmoid and
            # we unsigmoid the proposals before.

            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid

            topk = self.num_queries  # this is set to 900.

            # pick top k object queries based on the similarity with text.
            # TODO: think if this is the right thing to do and if there are any gotchas.
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # we just gather the topk boxes for each batch sample.
            ## This code is not DDP friendly, hence depreciating it
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )  # unsigmoid

            # pred boxes stops here, note, this is still unsigmoid.
            # proposals + deltas.
            # also note, these are the final queries that are passed to the decoder.
            refpoint_embed_ = refpoint_embed_undetach.detach()

            # this is not used anywhere, just returned.!!!.
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # this is not used anywhere, just returned!!!.
            # gather tgt
            tgt_undetach = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )

            # we set the tgt as the weights of embedding layer.
            # TODO: how is the embedding layer initialized or trained?
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            # this is useful if we always want to include some refpoints.
            # but we don't use it here.
            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(self.two_stage_type)
            )
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################

        # harshaL: hs -> output features after each layer.
        # references -> refined reference points after each layer.

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            blueglass_conf=blueglass_conf,
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################

        if self.two_stage_type == "standard":
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share
            )

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # beautiful implementation!
        # we use every pixel in a feature map as an anchor.
        # this is probably inspired from FCOS paper.
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # we generate *absolute* coordinates for
            # every pixel with center as the reference point.
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # then we convert absolute coords into
            # *relative* coords, detr uses relative coords.
            # note that, the relative coordinates *here* are with
            # respect to the actual image area H_valid, i.e.
            # exluding the padding area H = H_pad = H_valid + pad_size.
            # valid_ratios represent the ratio of image that is valid.
            # hence, H_valid can be computed as H * valid_ratio_H.
            # this is required so that attention only works with
            # refs of actual images and not pay attention to padding.
            # and also because the gt refs are computed wrt to the
            # H_valid and during loss computation we need both preds
            # and gt to be of the same scale.
            # note2, the reference points in padded area will have a
            # value > 1.0. this will be ignored during attention.
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        # we then stack all the reference point from all
        # the feature maps of all levels. Note because the sizes of
        # feature maps from different level levels are different
        # we scale them to a common scale by multiplying with valid ratios.
        # this also scales the refs to full image now as the image is
        # processed with the padding included.
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor | None = None,
        text_attention_mask: Tensor | None = None,
        pos_text: Tensor | None = None,
        text_self_attention_masks: Tensor | None = None,
        position_ids: Tensor | None = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        assert memory_text is not None, "expected memory_text"
        assert text_attention_mask is not None, "expected text_attention_mask."

        output = src

        # if num_layers = 0, there is no encoder.
        # this condition is probably unnecessary.

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        # here we generate the position encoding for
        # the text features to be used in the text enhancer layer.

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False
                )
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # this is the main encoding pipeline.
        # we have 6 layers of processing for encoding.
        # at each layer we first fuse the modalities using
        # image-to-text and text-to-image fusion module, refer,
        # BiAttentionBlock module (feature enhancer layer in paper).
        # we then process the text features through a transformer block.
        # and simultaneously process image features through a
        # deformable transformer block.
        # note, the only difference between text encoder and image encoder
        # at each level is image one is deformable whereas text is not.

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()

            if self.fusion_layers:
                if self.use_checkpoint:
                    ret = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                    assert (
                        ret is not None
                    ), "checkpointed function returned nothing. unexpected."
                    output, memory_text = ret
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []

        for layer_index, dec in enumerate(self.layers):
            dec.layer_index = layer_index

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        blueglass_conf: BLUEGLASSConf = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []

        # TODO: what are refpoints here.
        assert refpoints_unsigmoid is not None, "expected argument: refpoint_unsigmoid."
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        assert valid_ratios is not None, "expected argument: valid_ratios."
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[None, :]
                )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                blueglass_kwargs={
                    "blueglass_conf": blueglass_conf,
                    "reference_points": reference_points,
                },
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # harhsal: this basically refines the reference points by computing
            # deltas using attention mechanism from decoder queries + memory output
            # from the encoder. the reference points are computed from the memory features
            # and static proposals, and then refined here with text guided and everything.
            assert isinstance(memory_text, Tensor), "unexpected memory text."

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            if self.norm is not None:
                intermediate.append(self.norm(output))
            else:
                intermediate.append(output)

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MultiHeadAttnKnockOff(nn.MultiheadAttention):
    def knockoff_band(self, knockoff_band):
        """
        Knocking off columns along the colums following the residual dimensions
        """
        with torch.no_grad():
            self.in_proj_weight[:, knockoff_band] = 0
            if self.q_proj_weight is not None:
                self.q_proj_weight[:, [knockoff_band]] = 0
            if self.k_proj_weight is not None:
                self.k_proj_weight[:, [knockoff_band]] = 0
            if self.v_proj_weight is not None:
                self.v_proj_weight[:, [knockoff_band]] = 0
            if self.out_proj.bias is not None:
                self.out_proj.bias[knockoff_band] = 0

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        knockoff_band: list = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if knockoff_band is not None:
            self.knockoff_band(knockoff_band)
        attn_output, attn_output_weights = super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        return attn_output, attn_output_weights


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
        layer_index: int = None,
    ):
        super().__init__()
        self.layer_index = layer_index

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # Replaced self-attention with MultiHeadAttnKnockOff, enabling weight knockoff functionality
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn = MultiHeadAttnKnockOff(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

        assert self.layer_index is not None, "layer_index not set."
        layer_name = f"layer_{self.layer_index}"

        # DOCUMENT BLUEGLASS FEATURE HERE and repeat on all additions.
        intercept_manager().recorder(FeaturePattern.DET_DECODER_MLP).record(
            layer_name,
            {
                "pos_img": tgt2.detach().cpu(),
            },
        )

        pattern_name = FeaturePattern.DET_DECODER_MLP
        subpattern_name = FeatureSubPattern.POS_IMG
        name = f"{layer_name}.{pattern_name.value}.{subpattern_name.value}"
        tgt2 = intercept_manager().patcher(name).patch(name, tgt2)

        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        blueglass_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None
        assert memory is not None, "expected memory."
        assert blueglass_kwargs is not None, "blueglass_kwargs are None."

        base_path = "./SAE_RESULTS/SAE_RESULTS_w_bias/sae_result_conc_circ/L{}/L{}_EF{}/knockoff_filtered_data_points/tsne_rows_concentric_circles/indices_0_{}.npy"
        expansion_factor = 256
        circle_index = 50
        ## w bias
        # circle_indices = {0:90, 1:90, 2:50, 3:90, 4:50, 5:90}
        # circle_indices = {0:90, 1:50, 2:50, 3:50, 4:50, 5:50}
        ## wo bias
        # circle_indices = {0:90, 1:50, 2:90, 3:75, 4:75, 5:50}

        # knockoff_config = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
        # add_bias        = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}

        # filter_ind_knockoff = {0:L0_filter_ind_knockoff, 1:L1_filter_ind_knockoff, 2:L2_filter_ind_knockoff, 3:L3_filter_ind_knockoff, 4:L4_filter_ind_knockoff, 5:L5_filter_ind_knockoff}
        layer_knockoff_exp_config = blueglass_kwargs["blueglass_conf"][
            "layer_knock_off"
        ]
        knockoff = all(
            value is not None for value in layer_knockoff_exp_config.values()
        )

        reference_points = blueglass_kwargs["reference_points"].detach()

        layer_id = self.layer_index
        # knockoff = True and (True if layer_knockoff_exp_config is not None else False)
        tgt_orig = tgt
        knockoff_band = None
        if knockoff:
            circle_indices = layer_knockoff_exp_config.circle_indices
            circle_index = circle_indices[layer_id]
            knockoff_config = layer_knockoff_exp_config.knockoff_config
            if knockoff_config[layer_id]:
                L_filter_ind_knockoff = np.load(
                    base_path.format(layer_id, layer_id, expansion_factor, circle_index)
                )
                knockoff_band = L_filter_ind_knockoff
                with torch.no_grad():
                    tgt[:, :, L_filter_ind_knockoff] = 0
                    tgt_query_pos[:, :, L_filter_ind_knockoff] = 0

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2, attn_pattern = self.self_attn(
                q,
                k,
                tgt,
                attn_mask=self_attn_mask,
                average_attn_weights=False,
                knockoff_band=knockoff_band,
            )

            assert self.layer_index is not None, "layer_index not initialized."

            layer_name = f"layer_{self.layer_index}"
            intercept_manager().recorder(FeaturePattern.DET_DECODER_MHA).record(
                layer_name,
                {
                    "weights": attn_pattern.detach().cpu(),
                    "outputs": tgt2.detach().cpu(),
                },
            )

            pattern_name = FeaturePattern.DET_DECODER_MHA
            subpattern_name = FeatureSubPattern.OUTPUTS
            name = f"{layer_name}.{pattern_name.value}.{subpattern_name.value}"
            tgt2 = intercept_manager().patcher(name).patch(name, tgt2)

            assert self.dropout2 is not None, "dropout2 not initialized."
            assert self.norm2 is not None, "norm2 not initialized."
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]

            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
            knockoff_band=knockoff_band,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        layer_name = f"layer_{layer_id}"

        assert isinstance(tgt, Tensor), "unexpected type for feature."
        assert isinstance(tgt_reference_points, Tensor), "unexpected type for feature."
        intercept_manager().recorder(FeaturePattern.DET_DECODER_RESID_MHA).record(
            layer_name,
            {
                "pos_img": tgt.detach().cpu(),
                "refpnts": reference_points.cpu(),
            },
        )

        pattern_name = FeaturePattern.DET_DECODER_RESID_MHA
        subpattern_name = FeatureSubPattern.POS_IMG
        name = f"{layer_name}.{pattern_name.value}.{subpattern_name.value}"
        tgt = intercept_manager().patcher(name).patch(name, tgt)

        tgt = self.forward_ffn(tgt)

        assert isinstance(tgt, Tensor), "unexpected type for feature."
        intercept_manager().recorder(FeaturePattern.DET_DECODER_RESID_MLP).record(
            layer_name,
            {
                "pos_img": tgt.detach().cpu(),
                "refpnts": reference_points.cpu(),
            },
        )

        pattern_name = FeaturePattern.DET_DECODER_RESID_MLP
        subpattern_name = FeatureSubPattern.POS_IMG
        name = f"{layer_name}.{pattern_name.value}.{subpattern_name.value}"
        tgt = intercept_manager().patcher(name).patch(name, tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )
