# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
from enum import Enum
from typing import List, Tuple

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(os.getcwd(), "weights"))
FEATURE_DIR = os.environ.get("FEATURE_DIR", osp.join(os.getcwd(), "bluelens"))
MODELSTORE_CONFIGS_DIR = osp.join(os.getcwd(), "blueglass", "modeling", "modelstore")
MODELSTORE_MMDET_CONFIGS_DIR = osp.join(
    os.getcwd(), "blueglass", "third_party", "mmdet", "configs"
)


class Precision(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class Datasets(str, Enum):
    ECPERSONS_TRAIN = "ecpersons_train"
    ECPERSONS_VAL = "ecpersons_val"
    ECPERSONS_MINI = "ecpersons_mini"
    VALERIE22_TRAIN = "valerie22_train"
    VALERIE22_VAL = "valerie22_val"
    VALERIE22_MINI = "valerie22_mini"
    KITTI_TRAIN = "kitti_train"
    KITTI_VAL = "kitti_val"
    KITTI_MINI = "kitti_mini"
    BDD100K_TRAIN = "bdd100k_train"
    BDD100K_MINI = "bdd100k_mini"
    BDD100K_VAL = "bdd100k_val"
    LVIS_TRAIN = "lvis_train"
    LVIS_MINIVAL = "lvis_minival"
    LVIS_MINI = "lvis_mini"
    LVIS_VAL = "lvis_val"
    COCO_TRAIN = "coco_train"
    COCO_MINI = "coco_mini"
    COCO_VAL = "coco_val"
    FUNNYBIRDS_NO_INTERVENTION = "funnybirds_no_intervention"
    FUNNYBIRDS_NO_BEAK = "funnybirds_no_beak"
    FUNNYBIRDS_NO_EYES = "funnybirds_no_eyes"
    FUNNYBIRDS_NO_FOOT = "funnybirds_no_foot"
    FUNNYBIRDS_NO_TAIL = "funnybirds_no_tail"
    FUNNYBIRDS_NO_WINGS = "funnybirds_no_wings"
    FEATURES = "features"
    OPENIMAGES_TRAIN = "openimages_train"
    OPENIMAGES_VAL = "openimages_val"
    OPENIMAGES_MINI = "openimages_mini"
    OPENIMAGES_TRAIN_S0 = "openimages_train_s0"
    OPENIMAGES_TRAIN_S1 = "openimages_train_s1"
    OPENIMAGES_TRAIN_S2 = "openimages_train_s2"
    OPENIMAGES_TRAIN_S3 = "openimages_train_s3"
    OPENIMAGES_TRAIN_S4 = "openimages_train_s4"
    OPENIMAGES_TRAIN_S5 = "openimages_train_s5"
    OPENIMAGES_TRAIN_S6 = "openimages_train_s6"
    OPENIMAGES_TRAIN_S7 = "openimages_train_s7"
    OPENIMAGES_TRAIN_S8 = "openimages_train_s8"
    OPENIMAGES_TRAIN_S9 = "openimages_train_s9"
    OPENIMAGES_TRAIN_Sa = "openimages_train_sa"
    OPENIMAGES_TRAIN_Sb = "openimages_train_sb"
    OPENIMAGES_TRAIN_Sc = "openimages_train_sc"
    OPENIMAGES_TRAIN_Sd = "openimages_train_sd"
    OPENIMAGES_TRAIN_Se = "openimages_train_se"
    OPENIMAGES_TRAIN_Sf = "openimages_train_sf"


class Model(str, Enum):
    YOLO = "yolo"
    GDINO = "gdino"
    GENU = "genu"
    DINO_DETR = "dino_detr"
    DDETR = "ddetr"
    DETR = "detr"
    FLORENCE = "florence"
    PALIGEMMA = "paligemma"
    IDEFICS = "idefics"
    KOSMOS = "kosmos"
    INTERN = "intern"
    QWEN = "qwen"
    PHI = "phi"
    MINICPM = "minicpm"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    GPT_4O_MINI = "gpt_40_mini"
    CLAUDE = "claude"


class Evaluator(str, Enum):
    COCO = "coco"
    BDD100K = "bdd100k"
    LVIS = "lvis"


class Runner(str, Enum):
    MODELSTORE = "modelstore"
    LAYERS_PATCH = "layers_patch"
    FEATURE_EXTRACT = "feature_extract"
    VLM_LINEAR_PROBE = "vlm_linear_probe"
    SAE_LINEAR_PROBE = "sae_linear_probe"
    INTERPRETATION = "interpretation"
    SAE = "sae"
    DECODER_CLUSTER = "decoder_cluster"
    KNOCKOFF_LAYER = "knockoff_layer"


class FeaturePattern(str, Enum):
    ENCODER_LAYERS = "encoder_layers"
    DECODER_LAYERS = "decoder_layers"
    DET_DECODER_MHA = "det_decoder_mha"
    DET_DECODER_SA_MHA = "det_decoder_sa_mha"
    DET_DECODER_CA_MHA = "det_decoder_ca_mha"
    LLM_DECODER_MHA = "llm_decoder_mha"
    DET_DECODER_RESID_MHA = "det_decoder_resid_mha"
    DET_DECODER_CA_RESID_MHA = "det_decoder_ca_resid_mha"
    DET_DECODER_SA_RESID_MHA = "det_decoder_sa_resid_mha"
    LLM_DECODER_RESID_MHA = "llm_decoder_resid_mha"
    LLM_DECODER_MLP = "llm_decoder_mlp"
    DET_DECODER_MLP = "det_decoder_mlp"
    DET_DECODER_RESID_MLP = "det_decoder_resid_mlp"
    LLM_DECODER_RESID_MLP = "llm_decoder_resid_mlp"
    IO = "io"


class FeatureSubPattern(str, Enum):
    WEIGHTS = "weights"
    OUTPUTS = "outputs"

    PRE = "pre"
    POS = "pos"

    PRE_IMG = "pre_img"
    POS_IMG = "pos_img"
    PRE_TXT = "pre_txt"
    POS_TXT = "pos_txt"
    REFPNTS = "refpnts"


class SAEVariant(str, Enum):
    """
    For Relu: Sparsity loss coeff=0.01
    """

    RELU = "relu"
    TOPK = "topk"
    TOPK_FAST = "topk_fast"
    MATRYOSHKA = "matryoshka"
    SPECTRAL = "spectral"
    SPLINE = "spline"
    CROSSCODER = "crosscoder"


class ProbeVariant(str, Enum):
    BOX_LINEAR = "box_linear"
    CLS_LINEAR = "cls_linear"
    DET_LINEAR = "det_linear"


class Prompter(str, Enum):
    BASIC = "basic"
    ENSEMBLE = "ensemble"


class Matcher(str, Enum):
    EMBED_SIM = "embed_sim"
    LLM = "llm"


class Encoder(str, Enum):
    CLIP = "clip"
    NVEMBED = "nvembed"
    B1ADE = "b1ade"
    BERT = "bert"
    SIGLIP = "siglip"


class RunnerMode(str, Enum):
    TRAIN = "train"
    INFER = "infer"
    TEST = "test"


class InterceptMode(str, Enum):
    MANUAL = "manual"
    HOOKS = "hooks"


class Optimizer(str, Enum):
    ADAMW = "adamw"


class Scheduler(str, Enum):
    COSINE = "cosine"
    MULTISTEP = "multistep"


DATASETS_AND_EVALS: List[Tuple[str, Datasets, Datasets, Evaluator]] = [
    (
        "funnybirds",
        Datasets.FUNNYBIRDS_NO_INTERVENTION,
        Datasets.FUNNYBIRDS_NO_BEAK,
        Evaluator.COCO,
    ),
    ("valerie", Datasets.VALERIE22_TRAIN, Datasets.VALERIE22_VAL, Evaluator.COCO),
    ("ecpersons", Datasets.ECPERSONS_TRAIN, Datasets.ECPERSONS_VAL, Evaluator.COCO),
    ("kitti", Datasets.KITTI_TRAIN, Datasets.KITTI_VAL, Evaluator.COCO),
    ("bdd100k", Datasets.BDD100K_TRAIN, Datasets.BDD100K_VAL, Evaluator.BDD100K),
    ("coco", Datasets.COCO_TRAIN, Datasets.COCO_VAL, Evaluator.COCO),
    ("lvis", Datasets.LVIS_TRAIN, Datasets.LVIS_MINIVAL, Evaluator.LVIS),
]


DATASETS_AND_Extraction: List[Tuple[str, Datasets, Datasets, Evaluator]] = [
    (
        "funnybirds",
        Datasets.FUNNYBIRDS_NO_INTERVENTION,
        Datasets.FUNNYBIRDS_NO_BEAK,
        Evaluator.COCO,
    ),
    ("valerie", Datasets.VALERIE22_TRAIN, Datasets.VALERIE22_VAL, Evaluator.COCO),
    ("ecpersons", Datasets.ECPERSONS_TRAIN, Datasets.ECPERSONS_VAL, Evaluator.COCO),
    ("kitti", Datasets.KITTI_TRAIN, Datasets.KITTI_VAL, Evaluator.COCO),
    ("bdd100k", Datasets.BDD100K_TRAIN, Datasets.BDD100K_VAL, Evaluator.BDD100K),
    ("coco", Datasets.COCO_TRAIN, Datasets.COCO_VAL, Evaluator.COCO),
    ("lvis", Datasets.LVIS_TRAIN, Datasets.LVIS_MINIVAL, Evaluator.LVIS),
]
