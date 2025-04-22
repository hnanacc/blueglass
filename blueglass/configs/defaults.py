# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator, Union
import torch

from typing import Literal, Optional, List, Tuple, Any
from .constants import (
    Datasets,
    Model,
    Evaluator,
    Prompter,
    Matcher,
    Encoder,
    Runner,
    RunnerMode,
    Optimizer,
    Precision,
    Scheduler,
    SAEVariant,
    ProbeVariant,
    InterceptMode,
    FeaturePattern,
    FeatureSubPattern,
)


@dataclass
class DatasetConf:
    label: Datasets = Datasets.COCO_MINI
    train: Datasets = Datasets.COCO_MINI
    infer: Datasets = Datasets.COCO_MINI
    test: Datasets = Datasets.COCO_MINI

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

    max_steps: int = 100_000
    logs_period: int = 1
    eval_period: int = 1000
    patch_eval_period: int = 2000
    ckpt_period: int = 1000
    resume: bool = False
    save_ckpt_locally: bool = field(default=False)

    optimizer: Optimizer = Optimizer.ADAMW
    warmup_steps: int = 1
    grad_acc_steps: int = 1
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    eps: float = 5e-7
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    precision: Precision = Precision.FLOAT32

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

    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


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
    filter_column_scheme: Optional[Dict[str, Union[int, float, str]]] = None
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
