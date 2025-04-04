# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.configs import BLUEGLASSConf
from .autoencoder import AutoEncoder


class Crosscoder(AutoEncoder):
    def __init__(self, conf: BLUEGLASSConf, feature_in_dim: int):
        pass
