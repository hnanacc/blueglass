# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os.path as osp
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from blueglass.configs import *

from typing import Optional


@dataclass
class ProbeRunnerConf(RunnerConf):
    name: Runner = Runner.VLM_LINEAR_PROBE
    mode: RunnerMode = RunnerMode.TRAIN

    lr: float = 0.0001


@dataclass
class ProbeDatasetConf(DatasetConf):
    batch_size: int = 4


@dataclass
class ProbeFeatureConf(FeatureConf):
    path: Optional[str] = FEATURE_DIR
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    pattern: FeaturePattern = FeaturePattern.DET_DECODER_RESID_MLP
    subpattern: FeatureSubPattern = FeatureSubPattern.POS_IMG


def register_probes():
    cs = ConfigStore.instance()

    for variant in ProbeVariant:
        for ds_name, ds_train, ds_test, ev in DATASETS_AND_EVALS:
            cs.store(
                f"probe.{variant}.mmdet_dinodetr.{ds_name}",
                BLUEGLASSConf(
                    runner=ProbeRunnerConf(),
                    dataset=ProbeDatasetConf(
                        train=ds_train, test=ds_test, label=ds_test
                    ),
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
                    evaluator=EvaluatorConf(names=ev, use_multi_layer=True),
                    feature=ProbeFeatureConf(
                        path=osp.join(FEATURE_DIR, "mmdet_dinodetr")
                    ),
                    probe=ProbeConf(variant=variant, use_vlm_pred_as_true=True),
                    experiment=ExperimentConf(
                        name=f"probe_{variant}_mmdet_dinodetr_{ds_name}"
                    ),
                ),
            )

            cs.store(
                f"probe.{variant}.mmdet_detr.{ds_name}",
                BLUEGLASSConf(
                    runner=ProbeRunnerConf(),
                    dataset=ProbeDatasetConf(
                        train=ds_train, test=ds_test, label=ds_test
                    ),
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
                    evaluator=EvaluatorConf(names=ev, use_multi_layer=True),
                    feature=ProbeFeatureConf(path=osp.join(FEATURE_DIR, "mmdet_detr")),
                    probe=ProbeConf(variant=variant, use_vlm_pred_as_true=True),
                    experiment=ExperimentConf(
                        name=f"probe_{variant}_mmdet_etr_{ds_name}"
                    ),
                ),
            )

            cs.store(
                f"probe.{variant}.gdino.{ds_name}",
                BLUEGLASSConf(
                    runner=ProbeRunnerConf(),
                    dataset=ProbeDatasetConf(
                        train=ds_train, test=ds_test, label=ds_test
                    ),
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
                    evaluator=EvaluatorConf(names=ev, use_multi_layer=True),
                    feature=ProbeFeatureConf(path=osp.join(FEATURE_DIR, "gdino")),
                    probe=ProbeConf(variant=variant, use_vlm_pred_as_true=True),
                    experiment=ExperimentConf(name=f"probe_{variant}_gdino_{ds_name}"),
                ),
            )

            cs.store(
                f"probe.{variant}.genu.{ds_name}",
                BLUEGLASSConf(
                    runner=ProbeRunnerConf(),
                    dataset=ProbeDatasetConf(
                        train=ds_train, test=ds_test, label=ds_test
                    ),
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
                    evaluator=LabelMatchEvaluatorConf(names=ev, use_multi_layer=True),
                    feature=ProbeFeatureConf(path=osp.join(FEATURE_DIR, "genu")),
                    probe=ProbeConf(variant=variant, use_vlm_pred_as_true=True),
                    experiment=ExperimentConf(name=f"probe_{variant}_genu_{ds_name}"),
                ),
            )

            cs.store(
                f"probe.{variant}.florence.{ds_name}",
                BLUEGLASSConf(
                    runner=ProbeRunnerConf(),
                    dataset=ProbeDatasetConf(
                        train=ds_train, test=ds_test, label=ds_test
                    ),
                    model=ModelConf(name=Model.FLORENCE),
                    evaluator=LabelMatchEvaluatorConf(names=ev, use_multi_layer=True),
                    feature=ProbeFeatureConf(path=osp.join(FEATURE_DIR, "florence")),
                    probe=ProbeConf(variant=variant, use_vlm_pred_as_true=True),
                    experiment=ExperimentConf(
                        name=f"probe_{variant}_florence_{ds_name}"
                    ),
                ),
            )
