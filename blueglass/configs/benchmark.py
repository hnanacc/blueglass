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
    Runner,
    RunnerMode,
    Model,
)
from .constants import WEIGHTS_DIR, BENCHMARKS_DIR, DATASETS_AND_EVALS


@dataclass
class BenchmarkDatasetConf(DatasetConf):
    batch_size: int = 8


@dataclass
class BenchmarkRunnerConf(RunnerConf):
    name: Runner = Runner.BENCHMARK
    mode: RunnerMode = RunnerMode.TEST


def register_benchmarks():
    cs = ConfigStore.instance()

    for ds_name, _, ds_test, ev in DATASETS_AND_EVALS:
        cs.store(
            f"benchmark.yolo.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.YOLO,
                    checkpoint_path=osp.join(WEIGHTS_DIR, "yolo", "yolov8x-oiv7.pt"),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"benchmark_yolo_{ds_name}"),
            ),
        )

        cs.store(
            f"benchmark.dino.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.DINO,
                    conf_path=osp.join(
                        BENCHMARKS_DIR, "mmbench", "configs", f"dino_{ds_name}.py"
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "dino", f"finetuned_dino_{ds_name}.pt"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"benchmark_dino_{ds_name}"),
            ),
        )

        cs.store(
            f"benchmark.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GDINO,
                    conf_path=osp.join(
                        BENCHMARKS_DIR,
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
                experiment=ExperimentConf(name=f"benchmark_gdino_{ds_name}"),
            ),
        )

        cs.store(
            f"benchmark.genu.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GENU,
                    conf_path=osp.join(
                        BENCHMARKS_DIR,
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
                experiment=ExperimentConf(name=f"benchmark_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"benchmark.florence.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"benchmark_florence_{ds_name}"),
            ),
        )

        cs.store(
            f"benchmark.gemini.{ds_name}",
            BLUEGLASSConf(
                runner=BenchmarkRunnerConf(),
                dataset=BenchmarkDatasetConf(test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GEMINI,
                    api_key=os.getenv("GEMINI_KEY", None),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"benchmark_gemini_{ds_name}"),
            ),
        )
