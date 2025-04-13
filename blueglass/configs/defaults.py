# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from enum import Enum
from dataclasses import dataclass, field
import torch

from typing import Literal, Optional, List, Tuple, Any


class Precision(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class Dataset(str, Enum):
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
    BDD100k_VAL = "bdd100k_val"
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
    SAE = "sae"


class FeaturePattern(str, Enum):
    ENCODER_LAYERS = "encoder_layers"
    DECODER_LAYERS = "decoder_layers"
    DET_DECODER_MHA = "det_decoder_mha"
    LLM_DECODER_MHA = "llm_decoder_mha"
    DET_DECODER_RESID_MHA = "det_decoder_resid_mha"
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


@dataclass
class DatasetConf:
    label: Dataset = Dataset.COCO_MINI
    train: Dataset = Dataset.COCO_MINI
    infer: Dataset = Dataset.COCO_MINI
    test: Dataset = Dataset.COCO_MINI

    batch_size: int = 8


@dataclass
class ModelConf:
    name: Model = Model.GDINO
    hf_id: Optional[str] = None
    conf_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_path_genu_embed: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class EvaluatorConf:
    name: Evaluator = Evaluator.COCO
    use_label_matcher: bool = False
    use_multi_layer: bool = False
    use_descriptions: bool = False
    use_box_ious: bool = False
    use_box_objectness: bool = False
    use_negatives: bool = False
    use_parts: bool = False
    prompter: Prompter = Prompter.BASIC
    matcher: Matcher = Matcher.EMBED_SIM
    encoder: Encoder = Encoder.CLIP
    num_topk_matches: int = 1
    max_predictions: int = 900
    min_threshold_cls: float = 0.0
    min_threshold_box: float = 0.0
    compute_confusion: bool = False
    use_analysis: bool = False
    num_vis_samples: int = 0


@dataclass
class LabelMatchEvaluatorConf(EvaluatorConf):
    use_label_matcher: bool = True
    use_descriptions: bool = True
    max_predictions: int = 900
    min_threshold_cls: float = 0.0
    num_topk_matches: int = 1
    use_box_ious: bool = True
    use_box_objectness: bool = True


@dataclass
class RunnerConf:
    name: Runner = Runner.MODELSTORE
    mode: RunnerMode = RunnerMode.TEST

    max_steps: int = 50_000
    logs_period: int = 1
    eval_period: int = 100
    patch_eval_period: int = 200
    ckpt_period: int = 100
    resume: bool = False

    optimizer: Optimizer = Optimizer.ADAMW
    warmup_steps: int = 1
    grad_acc_steps: int = 1
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    eps: float = 5e-7
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    precision: Precision = Precision.BFLOAT16

    scheduler: Scheduler = Scheduler.MULTISTEP
    milestones: List[int] = field(default_factory=lambda: [40_000, 45_000, 49_000])


@dataclass
class SAEConf:
    variant: SAEVariant = SAEVariant.TOPK
    expansion_factor: int = 32

    use_feature_norm: bool = True
    use_feature_bias: bool = True
    use_latents_bias: bool = True
    use_decoder_norm: bool = True

    latents_topk: int = 32
    latents_topk_aux: int = 512
    group_multipliers: List[float] = field(default_factory=lambda: [1, 1, 2, 4, 8])

    loss_sparsity_coeff: float = 0.0
    loss_reconstr_coeff: float = 1.0
    loss_topk_aux_coeff: float = 1 / 32

    threshold_top_latents: float = 0.5
    threshold_update_rate: float = 0.01
    threshold_latents_dead: int = 1_00_000
    min_threshold_latents_dead: int = 10_000
    threshold_latents_dense: float = 0.5


@dataclass
class ProbeConf:
    variant: ProbeVariant = ProbeVariant.BOX_LINEAR
    fwd_period: int = 1
    use_vlm_pred_as_true: bool = True


@dataclass
class LayerKnockoffExpConf:
    top_irrelevant_idx: Optional[dict] = None
    knockoff_config: Optional[dict] = None
    irrelevant_idx_dir: Optional[str] = None


@dataclass
class FeatureConf:
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    patterns: List[FeaturePattern] = field(
        default_factory=lambda: [
            FeaturePattern.DET_DECODER_RESID_MLP,
            FeaturePattern.IO,
        ]
    )
    sub_patterns: List[FeatureSubPattern] = field(
        default_factory=lambda: [FeatureSubPattern.POS_IMG]
    )
    layer_ids: List[int] = field(default_factory=lambda: [])
    path: Optional[str] = None
    use_cached: bool = False

    batch_size: int = 512
    compute_confusion: bool = False
    max_rows_per_part: int = 500_000


@dataclass
class PatchConf:
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    patterns: List[FeaturePattern] = field(
        default_factory=lambda: [FeaturePattern.DET_DECODER_RESID_MLP]
    )
    layer_indx: List[Optional[int]] = field(default_factory=lambda: [None])
    suffix: Optional[str] = field(default_factory=lambda: None)
    sae_paths: List[Optional[str]] = field(default_factory=lambda: [None])
    use_wandb: bool = False


@dataclass
class ExperimentConf:
    name: str = field(default="test_experiment")
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "runs"))
    use_wandb: bool = field(default=True)
    wandb_every_proc: bool = field(default=False)
    wandb_project_name: str = field(default="blueglass")
    wandb_entity_name: str = field(default="blueglass_entity")
    seed: int = field(default=1337)


@dataclass
class BLUEGLASSConf:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"override hydra/job_logging": "none"},
            {"override hydra/hydra_logging": "none"},
        ]
    )
    hydra: Any = field(
        default_factory=lambda: {"output_subdir": None, "run": {"dir": "."}}
    )

    dataset: DatasetConf = field(default_factory=DatasetConf)
    model: ModelConf = field(default_factory=ModelConf)
    evaluator: EvaluatorConf = field(default_factory=EvaluatorConf)
    runner: RunnerConf = field(default_factory=RunnerConf)
    experiment: ExperimentConf = field(default_factory=ExperimentConf)
    feature: FeatureConf = field(default_factory=FeatureConf)
    patch: PatchConf = field(default_factory=PatchConf)
    sae: SAEConf = field(default_factory=SAEConf)
    probe: ProbeConf = field(default_factory=ProbeConf)
    layer_knock_off: LayerKnockoffExpConf = field(default_factory=LayerKnockoffExpConf)

    num_cpus: int = 16
    num_gpus: int = 1
    num_data_workers: int = 4
    num_machines: int = 1
    machine_rank: int = 0
    dist_url: str = "auto"
