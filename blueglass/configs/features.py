# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os.path as osp
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from .defaults import (
    BLUEGLASSConf,
    DatasetConf,
    ModelConf,
    RunnerConf,
    ExperimentConf,
    LabelMatchEvaluatorConf,
    FeatureConf,
    EvaluatorConf,
    LabelMatchEvaluatorConf,
    Runner,
    RunnerMode,
    InterceptMode,
    FeaturePattern,
    Model,
)
from .constants import (
    DATASETS_AND_EVALS,
    WEIGHTS_DIR,
    MODELSTORE_CONFIGS_DIR,
    MODELSTORE_MMDET_CONFIGS_DIR,
    FEATURE_DIR,
)

from typing import List, Optional


@dataclass
class ExtractRunnerConf(RunnerConf):
    name: Runner = Runner.FEATURE_EXTRACT
    mode: RunnerMode = RunnerMode.INFER
    logs_period: int = 1
    eval_period: int = 5


@dataclass
class ExtractDatasetConf(DatasetConf):
    batch_size: int = 12


@dataclass
class ExtractFeatureConf(FeatureConf):
    path: Optional[str] = FEATURE_DIR
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    patterns: List[FeaturePattern] = field(
        default_factory=lambda: [
            FeaturePattern.DET_DECODER_RESID_MLP,
            FeaturePattern.DET_DECODER_MLP,
            FeaturePattern.DET_DECODER_RESID_MHA,
            # FeaturePattern.DET_DECODER_MHA,
            FeaturePattern.IO,
        ]
    )


def register_features():
    cs = ConfigStore.instance()

    for ds_name, ds_train, _, ev in DATASETS_AND_EVALS:
        cs.store(
            f"features.mmdet_dinodetr.{ds_name}",
            BLUEGLASSConf(
                runner=ExtractRunnerConf(),
                dataset=ExtractDatasetConf(infer=ds_train, label=ds_train),
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
                feature=ExtractFeatureConf(),
                experiment=ExperimentConf(name=f"extract_dinodetr_{ds_name}"),
            ),
        )

        cs.store(
            f"features.mmdet_detr.{ds_name}",
            BLUEGLASSConf(
                runner=ExtractRunnerConf(),
                dataset=ExtractDatasetConf(infer=ds_train, label=ds_train),
                model=ModelConf(
                    name=Model.DETR,
                    conf_path=osp.join(
                        MODELSTORE_MMDET_CONFIGS_DIR,
                        "detr",
                        f"detr_r50_8xb2-150e_{ds_name}.py",
                    ),
                    checkpoint_path=osp.join(WEIGHTS_DIR, "detr", f"detr{ds_name}.pt"),
                ),
                evaluator=EvaluatorConf(name=ev),
                feature=ExtractFeatureConf(),
                experiment=ExperimentConf(name=f"extract_detr_{ds_name}"),
            ),
        )

        cs.store(
            f"features.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=ExtractRunnerConf(),
                dataset=ExtractDatasetConf(infer=ds_train, label=ds_train),
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
                feature=ExtractFeatureConf(),
                experiment=ExperimentConf(name=f"extract_gdino_{ds_name}"),
            ),
        )

        cs.store(
            f"features.genu.{ds_name}",
            BLUEGLASSConf(
                runner=ExtractRunnerConf(),
                dataset=ExtractDatasetConf(infer=ds_train, label=ds_train),
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
                        WEIGHTS_DIR, "genu", "lvis_v1_clip_a+cname_ViT-H.npy"
                    ),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                feature=ExtractFeatureConf(),
                experiment=ExperimentConf(name=f"features_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"features.florence.{ds_name}",
            BLUEGLASSConf(
                runner=ExtractRunnerConf(),
                dataset=ExtractDatasetConf(infer=ds_train, label=ds_train),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                feature=ExtractFeatureConf(),
                experiment=ExperimentConf(name=f"features_florence_{ds_name}"),
            ),
        )
