# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Optional, List, Union

from blueglass.configs import *

"""
top_irrelevant_idx
"""
top_irrelevant_idx = {0: 50, 1: 50, 2: 90, 3: 75, 4: 90, 5: 50}
# knockoff_layer_selection = {0: True, 1: True, 2: True, 3: True, 4: True, 5: True}


@dataclass
class saeknockoffExpConfig(LayerKnockoffExpConf):
    top_irrelevant_idx: dict = field(
        default_factory=lambda: {0: 50, 1: 50, 2: 90, 3: 75, 4: 90, 5: 50}
    )
    knockoff_layer_selection: dict = field(
        default_factory=lambda: {0: True, 1: True, 2: True, 3: True, 4: True, 5: True}
    )
    use_all_layers: Union[bool, str] = field(default_factory=lambda: "both")
    irrelevant_idx_dir = None
    knockoff_range: List[List[int]] = field(
        default_factory=lambda: [
            [0, 10],
            [0, 25],
            [0, 50],
            [0, 75],
            [0, 90],
            [0, 100],
            [100, 90],
            [100, 75],
            [100, 50],
            [100, 25],
            [100, 10],
        ]
    )


@dataclass
class SAEVariantConf(SAEConf):
    variant: SAEVariant = SAEVariant.TOPK_FAST
    topk: int = 32


@dataclass
class SaeKnockoffDatasetConf(DatasetConf):
    test_batch_size: int = 32


@dataclass
class SAEVariantConf(SAEConf):
    variant: SAEVariant = SAEVariant.TOPK_FAST
    topk: int = 32
    # loss_topk_aux_coeff: float = field(default_factory=lambda: 0)


@dataclass
class SAERunnerConf(RunnerConf):
    name: Runner = Runner.SAE
    mode: RunnerMode = RunnerMode.TRAIN
    lr: Optional[float] = field(default_factory=lambda: 1e-4)
    warmup_steps: int = 1
    eps: float = 1e-8
    precision: Precision = Precision.BFLOAT16


@dataclass
class SAEDatasetConf(DatasetConf):
    batch_size: int = 10


@dataclass
class SaeKnockoffRunnerConf(RunnerConf):
    name: Runner = Runner.SAE_KNOCKOFF
    mode: RunnerMode = RunnerMode.TRAIN


@dataclass
class SAEFeatureConf(FeatureConf):
    path: Optional[str] = FEATURE_DIR
    layer_ids: List = field(default_factory=lambda: [4, 5])
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    patterns: List[FeaturePattern] = field(
        default_factory=lambda: [
            # FeaturePattern.DET_DECODER_MHA,
            # FeaturePattern.DET_DECODER_MLP,
            FeaturePattern.DET_DECODER_RESID_MHA,
            FeaturePattern.DET_DECODER_RESID_MLP,
            FeaturePattern.IO,
        ]
    )
    use_cached: bool = True
    train_batch_size: int = 5000


def register_saeknockoff():
    cs = ConfigStore.instance()

    for ds_name, ds_train, ds_test, ev in DATASETS_AND_EVALS:
        cs.store(
            f"saeknockoff.yolo.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.YOLO,
                    checkpoint_path=osp.join(WEIGHTS_DIR, "yolo", "yolov8x-oiv7.pt"),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_yolo_{ds_name}"),
                layer_knock_off=saeknockoffExpConfig(),
            ),
        )

        cs.store(
            f"saeknockoff.mmdet_dinodetr.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
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
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(
                    name=f"knockoff_red_attn_wt_dinodetr_{ds_name}"
                ),
                layer_knock_off=saeknockoffExpConfig(),
            ),
        )

        cs.store(
            f"saeknockoff.mmdet_detr.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
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
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_detr_{ds_name}"),
                layer_knock_off=saeknockoffExpConfig(),
            ),
        )

        cs.store(
            f"saeknockoff.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
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
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_gdino_{ds_name}"),
                layer_knock_off=saeknockoffExpConfig(),
            ),
        )

        cs.store(
            f"saeknockoff.genu.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
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
                evaluator=LabelMatchEvaluatorConf(name=ev, num_topk_matches=3),
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"saeknockoff.florence.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(
                    name=f"knockoff_red_attn_wt_florence_{ds_name}"
                ),
            ),
        )

        cs.store(
            f"saeknockoff.gemini.{ds_name}",
            BLUEGLASSConf(
                runner=SaeKnockoffRunnerConf(),
                dataset=SaeKnockoffDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GEMINI,
                    api_key=os.getenv("GEMINI_KEY", None),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                sae=SAEVariantConf(),
                feature=SAEFeatureConf(),
                experiment=ExperimentConf(
                    name=f"knockoff_red_attn_wt_gemini_{ds_name}"
                ),
            ),
        )
