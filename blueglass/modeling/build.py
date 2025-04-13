# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
from torch import nn
from .modelstore.yolo.adaptation import YOLO_X
from .modelstore.grounding_dino.adaptation import GroundingDINO
from .modelstore.generateu.adaptation import GenerateU
from .modelstore.mmdet import MMDetModel
from .modelstore.huggingface import (
    PaliGemma2,
    Florence,
    IDEFICS2,
    Kosmos2,
    InternVL2,
    QwenVL,
    Phi3,
    DeepSeekVL,
)
from .modelstore.closed_source import Gemini, GPT4oMini, Claude
from blueglass.configs import BLUEGLASSConf, Model

logger = setup_blueglass_logger(__name__)


def build_model(conf: BLUEGLASSConf) -> nn.Module:
    model_classes = {
        Model.YOLO: YOLO_X,
        Model.GDINO: GroundingDINO,
        Model.GENU: GenerateU,
        Model.DINO_DETR: MMDetModel,
        Model.DETR: MMDetModel,
        Model.PALIGEMMA: PaliGemma2,
        Model.FLORENCE: Florence,
        Model.IDEFICS: IDEFICS2,
        Model.KOSMOS: Kosmos2,
        Model.INTERN: InternVL2,
        Model.QWEN: QwenVL,
        Model.PHI: Phi3,
        Model.DEEPSEEK: DeepSeekVL,
        Model.GEMINI: Gemini,
        Model.GPT_4O_MINI: GPT4oMini,
        Model.CLAUDE: Claude,
    }

    model_class = model_classes.get(conf.model.name)
    if model_class is None:
        raise NotImplementedError(f"Unsupported model: {conf.model.name}.")

    logger.debug(f"Building the model {conf.model.name}.")
    model = model_class(conf)
    logger.info(f"Successfully built the model {conf.model.name}.")
    return model
