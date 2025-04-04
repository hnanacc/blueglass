# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os


def setup_notebook(production=False):
    print("Setup paths.")

    root_path = os.path.expanduser("~/blueglass")
    assert os.path.exists(root_path), "cannot find the root path to blueglass."
    os.chdir(root_path)

    print("Import essential utils.")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from torch import nn, Tensor

    if production:
        import scienceplots

    import warnings

    warnings.filterwarnings("ignore")
