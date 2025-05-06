# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
from os import path as osp
import gc
import shutil
from functools import lru_cache
import torch
from torch import nn
from blueglass.configs import BLUEGLASSConf, Model
from blueglass.runners import Runner
from blueglass.utils.logger_utils import setup_blueglass_logger
from blueglass.configs import BLUEGLASSConf, FeaturePattern
from blueglass.features import build_feature_dataloader, FeatureDataset
from blueglass.modeling.saes import GroupedSAE
from blueglass.interpret import Interpreter, DatasetAttribution
from blueglass.configs.utils import load_blueglass_from_wandb

from IPython import embed

from typing import Dict, Any, Union


logger = setup_blueglass_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterpretationRunner(Runner):
    def __init__(self, conf: BLUEGLASSConf):
        super().__init__(conf)
        self.device = DEVICE
        self.conf = conf
        assert conf.sae.config_path is not None, "Require sae conf path."
        self.sae_conf = load_blueglass_from_wandb(
            conf.sae.config_path, original_config=self.conf
        )
        self.save_path = self.prepare_save_path(conf)
        self.interp_per_name = self.build_interpreters(self.sae_conf, self.save_path)

    def prepare_save_path(self, conf: BLUEGLASSConf):
        path = osp.join(conf.experiment.output_dir, "interp")
        if osp.exists(path):
            logger.info("Save path already exists. Overriden.")
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    def _frozen(self, model: nn.Module):
        for p in model.parameters():
            p.requires_grad = False
        return model.eval()

    def prepare_filter_scheme(
        self, conf: BLUEGLASSConf, remove_io: bool = True
    ) -> str:
        patterns = conf.feature.patterns
        if remove_io and FeaturePattern.IO in patterns:
            patterns.remove(FeaturePattern.IO)
        patterns = "|".join(patterns) if len(patterns) > 0 else r"\w+"

        subpatns = conf.feature.sub_patterns
        subpatns = "|".join(subpatns) if len(subpatns) > 0 else r"\w+"

        layerids = conf.feature.layer_ids
        layerids = [str(li) for li in layerids]
        layerids = "|".join(layerids) if len(layerids) > 0 else r"\d+"

        return f"layer_({layerids}).({patterns}).({subpatns})"
    
    def build_infer_dataloader(self, conf):
        return build_feature_dataloader(
            conf,
            conf.dataset.infer,
            conf.model.name,
            "test",
            self.prepare_filter_scheme(self.sae_conf),
        )

    @lru_cache()
    def prepare_metadata(self) -> Dict[str, Any]:
        return FeatureDataset(
            self.conf,
            self.conf.dataset.infer,
            self.conf.model.name,
            filter_scheme=self.prepare_filter_scheme(self.sae_conf),
        ).infer_feature_meta()

    def build_interpreters(
        self, conf: BLUEGLASSConf, save_path: str
    ) -> Dict[str, Interpreter]:
        metadata = self.prepare_metadata()
        return {
            name: DatasetAttribution(conf, feature_dim, osp.join(save_path, name))
            for name, feature_dim in metadata["feature_dim_per_name"].items()
        }

    def build_saes_model(self) -> nn.Module:
        """"
        Loads the SAEs from the different checkpoints and creates a model for infer/test mode while overriding the 
        blueglass config with the wandb config.
        The wandb config is used to load the correct model and the correct feature patterns.
        """

        filters = ["decoder_mlp"]
        filters  = []
        patterns = self.sae_conf.feature.patterns
        self.sae_conf.feature.patterns = [
            p for p in patterns if any(f in p.value for f in filters)
        ]
        
        metadata = FeatureDataset(
            self.sae_conf,
            self.sae_conf.dataset.infer,
            self.sae_conf.model.name,
            filter_scheme=self.prepare_filter_scheme(self.sae_conf),
        ).infer_feature_meta()

        assert (
            "feature_dim_per_name" in metadata
        ), "Feature dims not found in store meta."

        m = GroupedSAE(self.sae_conf, metadata["feature_dim_per_name"]).to(self.device)
        assert self.conf.sae.checkpoint_path is not None, "Require SAE checkpoint."
        ckpt = torch.load(
            self.conf.sae.checkpoint_path, map_location="cpu", weights_only=False
        )
        missing_keys, unexpected_keys = m.load_state_dict(ckpt["model"], strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logger.warning(
                f"Missing keys in state_dict: {missing_keys}. "
                f"Unexpected keys in state_dict: {unexpected_keys}."
            )
        return m
    
    def build_model(self, conf) -> nn.Module:
        
        model = self.build_saes_model()
        return model.eval().to(self.device)

    def run_step(self, batched_inputs_per_name: Dict[str, Any]):
        assert isinstance(
            self.model, GroupedSAE
        ), "Expected SAE model to be GroupedSAE."

        batched_outputs_per_name = self.model(
            batched_inputs_per_name, flatten_records=False
        )

        for name, batched_outputs in batched_outputs_per_name.items():
            self.interp_per_name[name].process(
                batched_inputs_per_name[name], batched_outputs
            )

    def infer(self):
        self.dataloader, self.model = self.initialize_infer_attrs()
        logger.info("Starting inference.")
        for self.step, data in enumerate(self.dataloader):
            records_dict = self.run_step(data)

            del records_dict
            torch.cuda.empty_cache()
            gc.collect()

            if self.step % self.logs_period == 0:
                logger.info(f"Processed {self.step} / {len(self.dataloader)}")

        for name, interp in self.interp_per_name.items():
            logger.info(f"Interpret for {name}.")
            interp.interpret()

    def train(self):
        raise NotImplementedError("Unsupported. use infer.")

    def test(self):
        raise NotImplementedError("Unsupported. use infer.")
