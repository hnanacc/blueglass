# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

"""Define logger and its format."""

import logging

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
