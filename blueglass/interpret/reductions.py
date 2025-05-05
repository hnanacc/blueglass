# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from torch import Tensor
from typing import List


def compute_effective_dimensions(features: List[Tensor]):
    pass


import os
import umap
from matplotlib import pyplot as plt
import scienceplots
from torch import Tensor
from typing import List, Optional, Literal

plt.style.use(["science", "ieee", "grid"])


def compute_umap(features: List[Tensor], labels: List[int]):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    embeds = reducer.fit_transform(features)
    return embeds[:, 0], embeds[:, 1]


def compute_tsne():
    pass


def compute_pca():
    pass


def visualize_clusters(
    features: List[Tensor],
    labels: List[int],
    save_path: str,
    return_map: bool = False,
    method: Literal["umap", "pca", "tsne"] = "umap",
):
    if method == "umap":
        xcoords, ycoords = compute_umap(features, labels)
    else:
        raise ValueError("invalid method.")

    plt.scatter(xcoords, ycoords, c=labels, s=10)
    plt.colorbar()
    plt.title(os.path.basename(save_path))
    plt.savefig(save_path)
