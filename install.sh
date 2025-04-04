# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

#!/usr/bin/bash

# -------------------------------------
# Setup environment.
# ---------------------------------

Define the CUDA installation path
CUDA_HOME=~/software/cuda-12.4.1

# Export environment variables
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

## Specify the path of your installed mamba
## Follow this to install mamba https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
MAMBA_PATH=~/micromamba
source "$MAMBA_PATH/etc/profile.d/mamba.sh"

echo "Setting up BLUE GLASS environment..."
echo "Set PROJECT_ROOT=$PWD"

set -exo pipefail

export PROJECT_ROOT=$PWD
export BENCHMARK_ROOT="$PROJECT_ROOT/blueglass/modeling/benchmarks"
export THIRD_PARTY_ROOT="$PROJECT_ROOT/blueglass/third_party"



# Verify if nvcc is found
echo "Using NVCC from: $(which nvcc)"
nvcc --version

# if command -v micromamba &> /dev/null; then
#     eval "$(micromamba shell hook --shell=$SHELL_TYPE)"
# else
#     echo "Require micromamba for installation."
#     exit 1
# fi


# echo "Need permission for installing system dependencies."
# sudo apt install g++ libgeos-dev environment-modules
# Try to install dependencies, but continue even if it fails
sudo apt install -y g++ libgeos-dev environment-modules || echo "Skipping installation due to error."

echo "Assumes CUDA_TOOLKIT >=12. Please make sure."
nvcc --version

echo "Creating a new environment: blueglass"
eval "$(micromamba shell.hook bash)"
micromamba create -n blueglass python=3.11 -y
micromamba activate blueglass

# -----
# Install common libraries and external depedencies.
# -----

pip install "pip==22.0.2" setuptools
pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
pip install \
    omegaconf \
    cloudpickle \
    fvcore \
    ftfy \
    tqdm \
    "git+https://github.com/openai/CLIP.git" \
    timm \
    transformers \
    datasets \
    black \
    ipython \
    ipykernel \
    ultralytics \
    more_itertools \
    wandb \
    accelerate \
    simple_parsing \
    natsort \
    einops \
    scipy \
    wandb \
    opencv-python \
    hydra-core \
    SciencePlots

pip install -U scalabel
# -----
# Install external libraries.
# -----

# Clone cocoapi into the 'coco' subfolder
COCO_DIR="$THIRD_PARTY_ROOT/coco"
if [ ! -d "$COCO_DIR" ]; then
    git clone https://github.com/ppwwyyxx/cocoapi.git "$COCO_DIR"
    echo "COCO repo cloned at $COCO_DIR"
else
    echo "COCO repo already exists at $COCO_DIR. Skipping clone."
fi
pip install -e "${THIRD_PARTY_ROOT}/coco/PythonAPI"
pip install -e "${COCO_DIR}/PythonAPI"
echo "COCO installed from source."

# Clone lvisapi into the 'lvis' subfolder
LVIS_DIR="$THIRD_PARTY_ROOT/lvis"
if [ ! -d "$LVIS_DIR" ]; then
    git clone https://github.com/lvis-dataset/lvis-api.git "$LVIS_DIR"
    echo "LVIS-API repo cloned at $LVIS_DIR"
    pip install -e "${THIRD_PARTY_ROOT}/lvis"
    echo "LVIS-API installed from source."
else
    echo "LVIS-API repo already exists at $LVIS_DIR. Skipping clone."
fi


# -----
# Prepare GenerateU environment.
# -----

if [ -d "${BENCHMARK_ROOT}/generateu" ]; then
    echo "Prepare GenerateU environment."

    rm -rf "${BENCHMARK_ROOT}/generateu/projects/DDETRS/ddetrs/models/deformable_detr/ops/MultiScaleDeformableAttention.egg-info"
    rm -rf "${BENCHMARK_ROOT}/generateu/projects/DDETRS/ddetrs/models/deformable_detr/ops/build"
    rm -rf "${BENCHMARK_ROOT}/generateu/projects/DDETRS/ddetrs/models/deformable_detr/ops/dist"
    rm -rf "${BENCHMARK_ROOT}/generateu/build"
    rm -rf "${BENCHMARK_ROOT}/generateu/detectron2.egg-info"

    pip install -e "${BENCHMARK_ROOT}/generateu"
    pip install -r "${BENCHMARK_ROOT}/generateu/requirements.txt"

    cd "${BENCHMARK_ROOT}/generateu/projects/DDETRS/ddetrs/models/deformable_detr/ops/"
    bash make.sh
    cd $PROJECT_ROOT

    echo "Done."
else
    echo "Project GenerateU doesn't exists. Skipping!"
fi

# -----
# Prepare Grounding DINO environment.
# -----

if [ -d "${BENCHMARK_ROOT}/grounding_dino" ]; then
    echo "Prepare Grounding DINO environment."

    rm -rf "${BENCHMARK_ROOT}/grounding_dino/build"
    rm -rf "${BENCHMARK_ROOT}/grounding_dino/groundingdino.egg-info"
    pip install -e "${BENCHMARK_ROOT}/grounding_dino"

    echo "Done."
else
    echo "Project Grounding DINO doesn't exists. Skipping!"
fi

# -----
# Prepare HFBench environment.
# -----

if [ -d "${BENCHMARK_ROOT}/huggingface" ]; then
    echo "Prepare HF Bench environment."
    pip install \
        google-genai \
        openai \
    
    echo "Done."
else
    echo "Project HFBench doesn't exists. Skipping!"
fi

# -----
# Prepare MMBench environment.
# -----


MMDET_DIR="$THIRD_PARTY_ROOT/mmdet"
rm -rf $MMDET_DIR
git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
echo "MMDET repo cloned at $MMDET_DIR"
echo "Prepare MM Bench environment."
pip install -U openmim
mim install mmengine
mim install mmcv

# 🔧 Patch __init__.py BEFORE installing
TARGET_FILE="$MMDET_DIR/mmdet/__init__.py"
if [ -f "$TARGET_FILE" ]; then
    echo "Patching mmcv version check in $TARGET_FILE..."
    sed -i 's/mmcv_version < digit_version(mmcv_maximum_version))/mmcv_version <= digit_version(mmcv_maximum_version))/' "$TARGET_FILE"
else
    echo "ERROR: Target file not found: $TARGET_FILE"
    echo "Please locate the correct '__init__.py' and manually replace:"
    echo "    mmcv_version < digit_version(mmcv_maximum_version))"
    echo "with:"
    echo "    mmcv_version <= digit_version(mmcv_maximum_version))"
    exit 1
fi

pip install -e $MMDET_DIR
echo "Done."

# pip messes up few things, fix them.
pip install --upgrade shapely
pip install --upgrade transformers


echo "Installing BLUE GLASS as a package"
pip install -e .