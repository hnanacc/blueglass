# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
from .bdd100k import register_bdd100k
from .kitti import register_kitti
from .coco import register_coco
from .lvis import register_lvis
from .ecpersons import register_ecpersons
from .valerie22 import register_valerie22
from .funnybirds import register_funnybirds
from .openimages import register_openimages

logger = setup_blueglass_logger(__name__)


def register_custom_datasets(args):
    def safe_register(register_function, name):
        try:
            register_function(args)
            logger.info(f"Successfully registered {name} dataset.")
        except Exception as e:
            logger.warning(f"Failed to register {name} dataset: {e}")

    safe_register(register_kitti, "KITTI")
    safe_register(register_coco, "COCO")
    safe_register(register_lvis, "LVIS")
    safe_register(register_bdd100k, "BDD100K")
    safe_register(register_ecpersons, "ECPersons")
    safe_register(register_valerie22, "Valerie22")
    safe_register(register_funnybirds, "FunnyBirds")
    safe_register(register_openimages, "OpenImages")
