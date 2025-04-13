# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from enum import Enum


class FeaturePatterns(str, Enum):
    ENCODER_FEATURES = "encoder_features"
    DECODER_FEATURES = "decoder_features"

    ENCODER_ATTENTION = "encoder_attention"
    DECODER_ATTENTION = "decoder_attention"

    SAE_ENCODER_LAYER_5 = "sae_encoder_layer_5"
    SAE_DECODER_LAYER_5 = "sae_decoder_layer_5"

    FINAL_LAYERS = "final_layers"
    FINAL_LAYERS_BBOX_EMBED = "final_layers_bbox_embed"
    FINAL_LAYERS_CLASS_EMBED = "final_layers_class_embed"


class Models(str, Enum):
    GDINO = "gdino"
    GENU = "genu"
    LLAVA = "llava"
    HFBENCH = "hfbench"
    MMDET = "mmdet"


class Datasets(str, Enum):
    FUNNYBIRDS_NO_INTERVENTION = "funnybirds_no_intervention"
    FUNNYBIRDS_NO_BEAK = "funnybirds_no_beak"
    FUNNYBIRDS_NO_WINGS = "funnybirds_no_wings"
    FUNNYBIRDS_NO_EYES = "funnybirds_no_eyes"
    FUNNYBIRDS_NO_LEGS = "funnybirds_no_legs"
    FUNNYBIRDS_NO_TAIL = "funnybirds_no_tail"

    COCO_TRAIN = "coco_train"
    COCO_MINI = "coco_mini"
    COCO_VAL = "coco_val"

    BDD100K_TRAIN = "bdd100k_train"
    BDD100K_MINI = "bdd100k_mini"
    BDD100K_VAL = "bdd100k_val"

    LVIS_TRAIN = "lvis_train"
    LVIS_MINIVAL = "lvis_minival"
    LVIS_MINI = "lvis_mini"
    LVIS_VAL = "lvis_val"

    ECPERSONS_TRAIN = "ecpersons_train"
    ECPERSONS_MINI = "ecpersons_mini"
    ECPERSONS_VAL = "ecpersons_val"

    VALERIE22_TRAIN = "valerie22_train"
    VALERIE22_MINI = "valerie22_mini"
    VALERIE22_VAL = "valerie22_val"
