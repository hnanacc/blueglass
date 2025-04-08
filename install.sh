# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

#!/bin/bash
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    SUDO='sudo'
else
    SUDO=''
fi

# Check if the environment is valid
./check_env.sh

result=$?
if [ $result -ne 0 ]; then
    echo "❌ Environment check failed. Please fix the issues and try again."
    exit 1
else
    echo "✅ Environment check passed."
fi


export PROJECT_ROOT=$PWD
export BENCHMARK_ROOT="$PROJECT_ROOT/blueglass/modeling/benchmarks"
export THIRD_PARTY_ROOT="$PROJECT_ROOT/blueglass/third_party"

# Create micromamba env
eval "$(micromamba shell hook --shell bash)"

# Check if existing micromamba environment exists
if micromamba env list | grep -q "blueglass_env"; then
    echo "⚠️  Environment 'blueglass_env' already exists. Skipping creation."
else
    echo "Creating micromamba environment 'blueglass_env'..."
    micromamba create -y -n blueglass_env python=3.11
fi
micromamba activate blueglass_env

# Install common python dependencies
pip install -r requirements.txt

# Clone and set up COCO API
echo "Clone and set up COCO API"
COCO_DIR="$THIRD_PARTY_ROOT/coco"
if [ ! -d "$COCO_DIR" ]; then
    git clone https://github.com/ppwwyyxx/cocoapi.git "$COCO_DIR"
    echo "COCO repo cloned at $COCO_DIR"
    pip install -e "${THIRD_PARTY_ROOT}/coco/PythonAPI"
    echo "COCO installed from source."
else
    echo "COCO repo already exists at $COCO_DIR. Skipping setup."
fi

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
    # pip install -r "${BENCHMARK_ROOT}/generateu/requirements.txt"

    if command -v nvidia-smi &> /dev/null; then
        echo "Building CUDA extensions for GenerateU..."
        cd "${BENCHMARK_ROOT}/generateu/projects/DDETRS/ddetrs/models/deformable_detr/ops/"
        bash make.sh
        cd $PROJECT_ROOT
    else
        echo "⚠️  No GPU detected. Skipping CUDA build for GenerateU."
    fi
    echo "Done."
else
    echo "⚠️  Project GenerateU doesn't exists. Skipping!"
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

echo "🔧 Patch __init__.py BEFORE installing"
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
echo "✅ Environment setup complete!"

pip install -e .