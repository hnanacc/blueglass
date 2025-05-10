# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os.path as osp
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Dict, Any, Optional, List, Iterator, Union
from blueglass.configs import *

from typing import List, Optional


@dataclass
class DecoderClusterSAEVariantConf(SAEConf):
    variant: SAEVariant = SAEVariant.TOPK_FAST
    topk: int = 32
    config_path: str = (
        "/home/squtub/github_repos/blueglass/trained_saes/config_exp128.json"
    )
    checkpoint_path: str = (
        "/home/squtub/github_repos/blueglass/trained_saes/config_exp128_4000.pth"  #
    )
    # loss_topk_aux_coeff: float = field(default_factory=lambda: 0)


@dataclass
class DecoderClusterRunnerConf(RunnerConf):
    name: Runner = Runner.DECODER_CLUSTER
    mode: RunnerMode = RunnerMode.INFER
    cluster_method: str = "hdbscan"
    n_clusters: int = 32
    max_steps: int = n_clusters + 1


@dataclass
class DecoderClusterDatasetConf(DatasetConf):
    batch_size: int = 10


@dataclass
class DecoderClusterFeatureConf(FeatureConf):
    path: Optional[str] = FEATURE_DIR
    layer_ids: List = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    intercept_mode: InterceptMode = InterceptMode.MANUAL
    patterns: List[FeaturePattern] = field(
        default_factory=lambda: [
            # FeaturePattern.DET_DECODER_MHA,
            FeaturePattern.DET_DECODER_MLP,
            FeaturePattern.DET_DECODER_RESID_MHA,
            FeaturePattern.DET_DECODER_RESID_MLP,
            FeaturePattern.IO,
        ]
    )
    use_cached: bool = True
    filter_column_scheme: Optional[Dict[str, Union[int, float, str]]] = field(
        default_factory=lambda: {"conf_msk": 1}
    )
    batch_size: int = 5000


def register_decoder_cluster():
    cs = ConfigStore.instance()

    for ds_name, ds_train, ds_test, ev in DATASETS_AND_EVALS:

        cs.store(
            f"decodercluster.gdino.{ds_name}",
            BLUEGLASSConf(
                runner=DecoderClusterRunnerConf(),
                dataset=DecoderClusterDatasetConf(
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
                evaluator=EvaluatorConf(name=ev),
                feature=DecoderClusterFeatureConf(),
                sae=DecoderClusterSAEVariantConf(),
                experiment=ExperimentConf(name=f"decodercluster_gdino_{ds_name}"),
            ),
        )
