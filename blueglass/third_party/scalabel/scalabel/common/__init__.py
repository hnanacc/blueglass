# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

"""Common python utilities."""

from . import logger
from .quiet import is_quiet, mute

__all__ = ["mute", "is_quiet"]
