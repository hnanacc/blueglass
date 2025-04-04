# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="blueglass",
    version="1.0",
    packages=find_packages(
        include=[
            "blueglass",
        ]
    ),
    package_dir={"blueglass": "blueglass"},
)

## install the blueglass package using pip install -e .
