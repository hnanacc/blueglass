# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="blueglass",
    packages=find_packages(include=["blueglass", "blueglass.*"]),
)
