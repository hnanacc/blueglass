# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os.path as osp
from dataclasses import dataclass
from typing import get_args
from hydra.core.config_store import ConfigStore
from .defaults import (
    BLUEGLASSConf,
    DatasetConf,
    ModelConf,
    EvaluatorConf,
    LabelMatchEvaluatorConf,
    RunnerConf,
    ExperimentConf,
    FeatureConf,
    SAEConf,
    SAEVariant,
    Model,
    Runner,
    RunnerMode,
    InterceptMode,
    FeaturePattern,
    FeatureSubPattern,
)
from .constants import (
    WEIGHTS_DIR,
    MODELSTORE_CONFIGS_DIR,
    MODELSTORE_MMDET_CONFIGS_DIR,
    FEATURE_DIR,
    DATASETS_AND_EVALS,
)


@dataclass
class SAERunnerConf(RunnerConf):
    name: Runner = Runner.SAE
    mode: RunnerMode = RunnerMode.TRAIN

    lr: float = 0.0001


@dataclass
class SAEDatasetConf(DatasetConf):
    batch_size: int = 4


@dataclass
class SAEFeatureConf(FeatureConf):
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    pattern: FeaturePattern = FeaturePattern.DET_DECODER_RESID_MLP
    subpattern: FeatureSubPattern = FeatureSubPattern.POS_IMG
    use_cached: bool = True


def register_layerpatch():
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
                        WEIGHTS_DIR, "dinodetr", f"dinodetr_{ds_name}.pt"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                feature=SAEFeatureConf(path=osp.join(FEATURE_DIR, "dinodetr")),
                sae=SAEConf(),
                experiment=ExperimentConf(name=f"layerpatch_dinodetr_{ds_name}"),
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
                    checkpoint_path=osp.join(WEIGHTS_DIR, "detr", f"detr_{ds_name}.pt"),
                ),
                evaluator=EvaluatorConf(name=ev),
                feature=SAEFeatureConf(path=osp.join(FEATURE_DIR, "detr")),
                sae=SAEConf(),
                experiment=ExperimentConf(name=f"layerpatch_detr_{ds_name}"),
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
                feature=SAEFeatureConf(path=osp.join(FEATURE_DIR, "gdino")),
                sae=SAEConf(),
                experiment=ExperimentConf(name=f"layerpatch_gdino_{ds_name}"),
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
                feature=SAEFeatureConf(path=osp.join(FEATURE_DIR, "genu")),
                sae=SAEConf(),
                experiment=ExperimentConf(name=f"layerpatch_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"saes.florence.{ds_name}",
            BLUEGLASSConf(
                runner=SAERunnerConf(),
                dataset=SAEDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                feature=SAEFeatureConf(path=osp.join(FEATURE_DIR, "florence")),
                sae=SAEConf(),
                experiment=ExperimentConf(name=f"layerpatch_florence_{ds_name}"),
            ),
        )
