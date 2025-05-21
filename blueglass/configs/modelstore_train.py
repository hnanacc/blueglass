# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import os.path as osp
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from blueglass.configs import *


@dataclass
class ModelstoreDatasetConf(DatasetConf):
    train_batch_size: int = 8
    test_batch_size: int = 8


@dataclass
class ModelStoreRunnerConf(RunnerConf):
    name: Runner = Runner.MODELSTORE
    mode: RunnerMode = RunnerMode.TRAIN
    logs_period: int = 1
    eval_period: int = 5


def register_modelstore_train():
    cs = ConfigStore.instance()

    for ds_name, ds_train, ds_test, ev in DATASETS_AND_EVALS:
        cs.store(
            f"modelstore_train.yolo.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.YOLO,
                    checkpoint_path=osp.join(WEIGHTS_DIR, "yolo", "yolov8x-oiv7.pt"),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"modelstore_yolo_{ds_name}"),
            ),
        )

        cs.store(
            f"modelstore_train.mmdet_dinodetr.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.DINO_DETR,
                    conf_path=osp.join(
                        MODELSTORE_MMDET_CONFIGS_DIR,
                        "dino",
                        f"dino-4scale_r50_improved_8xb2-12e_{ds_name}.py",
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "mmdet", "dinodetr", f"dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"modelstore_dinodetr_{ds_name}"),
            ),
        )

        cs.store(
            f"modelstore_train.mmdet_detr.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.DETR,
                    conf_path=osp.join(
                        MODELSTORE_MMDET_CONFIGS_DIR,
                        "detr",
                        f"detr_r50_8xb2-150e_{ds_name}.py",
                    ),
                    checkpoint_path=osp.join(
                        WEIGHTS_DIR, "mmdet", "detr", f"detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"
                    ),
                ),
                evaluator=EvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"modelstore_detr_{ds_name}"),
            ),
        )

        cs.store(
            f"modelstore_train.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
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
                experiment=ExperimentConf(name=f"modelstore_gdino_{ds_name}"),
            ),
        )

        cs.store(
            f"modelstore_train.genu.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
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
                experiment=ExperimentConf(name=f"modelstore_genu_{ds_name}"),
            ),
        )

        cs.store(
            f"modelstore_train.florence.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(name=Model.FLORENCE),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"modelstore_florence_{ds_name}"),
            ),
        )

        cs.store(
            f"modelstore_train.gemini.{ds_name}",
            BLUEGLASSConf(
                runner=ModelStoreRunnerConf(),
                dataset=ModelstoreDatasetConf(train=ds_train, test=ds_test, label=ds_test),
                model=ModelConf(
                    name=Model.GEMINI,
                    api_key=os.getenv("GEMINI_KEY", None),
                ),
                evaluator=LabelMatchEvaluatorConf(name=ev),
                experiment=ExperimentConf(name=f"modelstore_gemini_{ds_name}"),
            ),
        )
