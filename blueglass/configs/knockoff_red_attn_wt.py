# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from .defaults import (
    BLUEGLASSConf,
    DatasetConf,
    ModelConf,
    RunnerConf,
    ExperimentConf,
    EvaluatorConf,
    LabelMatchEvaluatorConf,
    LayerKnockoffExpConf,
    Runner,
    RunnerMode,
    Model,
)
from .constants import WEIGHTS_DIR, MODELSTORE_DIR, DATASETS_AND_EVALS

"""
top_irrelevant_idx
"""
top_irrelevant_idx = {0: 50, 1: 50, 2: 90, 3: 75, 4: 90, 5: 50}
knockoff_config = {0: True, 1: True, 2: True, 3: True, 4: True, 5: True}
layer_knockoff_exp_config = LayerKnockoffExpConf(
    top_irrelevant_idx=top_irrelevant_idx,
    knockoff_config=knockoff_config,
    irrelevant_idx_dir=None,
)


@dataclass
class BenchmarkDatasetConf(DatasetConf):
    batch_size: int = 1


@dataclass
class BenchmarkRunnerConf(RunnerConf):
    name: Runner = Runner.BENCHMARK
    mode: RunnerMode = RunnerMode.TEST


def register_layerknockoff():
    cs = ConfigStore.instance()

    for ds_name, _, ds_test, ev in DATASETS_AND_EVALS:
        cs.store(
            f"layerknockoff.yolo.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.YOLO,
                    checkpoint_path=osp.join(WEIGHTS_DIR, "yolo", "yolov8x-oiv7.pt"),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_yolo_{ds_name}"),
                layer_knock_off=layer_knockoff_exp_config,
            ),
        )

        cs.store(
            f"layerknockoff.dino.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.DINO,
                    conf_path=osp.join(
                        MODELSTORE_DIR, "mmbench", "configs", f"dino_{ds_name}.py"
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "dino", f"finetuned_dino_{ds_name}.pt"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_dino_{ds_name}"),
                layer_knock_off=layer_knockoff_exp_config,
            ),
        )

        cs.store(
            f"layerknockoff.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GDINO,
                    conf_path=osp.join(
                        MODELSTORE_DIR,
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
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_gdino_{ds_name}"),
                layer_knock_off=layer_knockoff_exp_config,
            ),
        )

        cs.store(
            f"layerknockoff.genu.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GENU,
                    conf_path=osp.join(
                        MODELSTORE_DIR,
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
                experiment=ExperimentConf(name=f"knockoff_red_attn_wt_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"layerknockoff.florence.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(
                    name=f"knockoff_red_attn_wt_florence_{ds_name}"
                ),
            ),
        )

        cs.store(
            f"layerknockoff.gemini.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GEMINI,
                    api_key=os.getenv("GEMINI_KEY", None),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(
                    name=f"knockoff_red_attn_wt_gemini_{ds_name}"
                ),
            ),
        )
