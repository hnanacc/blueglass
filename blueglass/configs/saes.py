# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os.path as osp
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from blueglass.configs.constants import (
    FEATURE_DIR,
    DATASETS_AND_EVALS,
    MODELSTORE_MMDET_CONFIGS_DIR,
    MODELSTORE_CONFIGS_DIR,
    WEIGHTS_DIR,
)

from blueglass.configs import (
    SAEConf,
    RunnerConf,
    ExperimentConf,
    DatasetConf,
    ModelConf,
    FeatureConf,
    EvaluatorConf,
    SAEVariant,
    Model,
    Runner,
    Precision,
    RunnerMode,
    InterceptMode,
    FeaturePattern,
    BLUEGLASSConf,
    LabelMatchEvaluatorConf,
)
from typing import List, Optional


@dataclass
class SAEVariantConf(SAEConf):
    variant: SAEVariant = SAEVariant.TOPK_FAST
    topk: int = 32
    # loss_topk_aux_coeff: float = field(default_factory=lambda: 0)


@dataclass
class SAERunnerConf(RunnerConf):
    name: Runner = Runner.SAE
    mode: RunnerMode = RunnerMode.TRAIN
    lr: Optional[float] = field(default_factory=lambda: None)
    warmup_steps: int = 1
    eps: float = 1e-8
    precision: Precision = Precision.BFLOAT16


@dataclass
class SAEDatasetConf(DatasetConf):
    batch_size: int = 10


@dataclass
class SAEFeatureConf(FeatureConf):
    path: Optional[str] = FEATURE_DIR
    layer_ids: List = field(default_factory=lambda: [4, 5])
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    patterns: List[FeaturePattern] = field(
        default_factory=lambda: [
            # FeaturePattern.DET_DECODER_MHA,
            # FeaturePattern.DET_DECODER_MLP,
            # FeaturePattern.DET_DECODER_RESID_MHA,
            FeaturePattern.DET_DECODER_RESID_MLP,
            FeaturePattern.IO,
        ]
    )
    use_cached: bool = True
    batch_size: int = 16000


def register_saes():
    cs = ConfigStore.instance()

    for ds_name, ds_train, ds_test, ev in DATASETS_AND_EVALS:
        cs.store(
            f"saes.mmdet_dinodetr.{ds_name}",
            BLUEGLASSConf(
                runner=SAERunnerConf(),
                dataset=SAEDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.DINO_DETR,
                    conf_path=osp.join(
                        MODELSTORE_MMDET_CONFIGS_DIR,
                        "dino",
                        f"dino-4scale_r50_improved_8xb2-12e_{ds_name}.py",
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "mmdet", "dinodetr", f"dinodetr_{ds_name}.pt"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                feature=SAEFeatureConf(),
                sae=SAEVariantConf(),
                experiment=ExperimentConf(name=f"sae_mmdet_dinodetr_{ds_name}"),
            ),
        )

        cs.store(
            f"saes.mmdet_detr.{ds_name}",
            BLUEGLASSConf(
                runner=SAERunnerConf(),
                dataset=SAEDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.DETR,
                    conf_path=osp.join(
                        MODELSTORE_MMDET_CONFIGS_DIR,
                        "detr",
                        f"detr_r50_8xb2-150e_{ds_name}.py",
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "mmdet", "detr", f"detr_{ds_name}.pt"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                feature=SAEFeatureConf(),
                sae=SAEVariantConf(),
                experiment=ExperimentConf(name=f"sae_mmdet_detr_{ds_name}"),
            ),
        )

        cs.store(
            f"saes.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=SAERunnerConf(),
                dataset=SAEDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GDINO,
                    conf_path=osp.join(
                        MODELSTORE_CONFIGS_DIR,
                        "grounding_dino",
                        "groundingdino",
                        "config",
                        "GroundingDINO_SwinT_OGC.py",
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "gdino", "groundingdino_swint_ogc.pth"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                feature=SAEFeatureConf(),
                sae=SAEVariantConf(),
                experiment=ExperimentConf(name=f"sae_gdino_{ds_name}"),
            ),
        )

        cs.store(
            f"saes.genu.{ds_name}",
            BLUEGLASSConf(
                runner=SAERunnerConf(),
                dataset=SAEDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GENU,
                    conf_path=osp.join(
                        MODELSTORE_CONFIGS_DIR,
                        "generateu",
                        "projects",
                        "DDETRS",
                        "configs",
                        "vg_grit5m_swinL.yaml",
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "genu", "vg_grit5m_swinL.pth"
                    ),
                    checkpoint_path_genu_embed=osp.join(
                        WEIGHTS_DIR, "lvis_v1_clip_a+cname_ViT-H.npy"
                    ),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                feature=SAEFeatureConf(),
                sae=SAEVariantConf(),
                experiment=ExperimentConf(name=f"sae_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"saes.florence.{ds_name}",
            BLUEGLASSConf(
                runner=SAERunnerConf(),
                dataset=SAEDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                feature=SAEFeatureConf(),
                sae=SAEVariantConf(),
                experiment=ExperimentConf(name=f"sae_florence_{ds_name}"),
            ),
        )
