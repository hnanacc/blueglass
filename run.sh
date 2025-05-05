#!/usr/bin/bash
# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

# -------------------------------------
# Setup Variables Here.
# -------------------------------------

export TOKENIZERS_PARALLELISM=false
export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
export CUDA_VISIBLE_DEVICES=0

export DATASET_DIR="/nwstore/datasets"
export WEIGHTS_DIR="/nwstore/harshal/weights"
export FEATURE_DIR="/nwstore/datasets/BlueLens"


# -------------------------------------
# Setup Environment.
# -------------------------------------

MAMBA_PATH=~/micromamba
source "$MAMBA_PATH/etc/profile.d/mamba.sh"

if command -v micromamba &> /dev/null; then
    echo "✅ micromamba version: $(micromamba --version)"
else
    echo "No environment manager not found. Cannot proceed."
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate blueglass_env

echo "Using PYTHON : $(which python)"
nvidia-smi

# BATCH_SIZE=131072
# BATCH_SIZE=57344
# BATCH_SIZE=49152
# BATCH_SIZE=32768
BATCH_SIZE=1024

echo "World size: $WORLD_SIZE"
echo "Ceiled per-worker batch size: $BATCH_SIZE"

SAE_VARIANT=TOPK_FAST
DATASET="COCO"
EXPERIMENT_NAME="sae_interp_topk_fast_32"
python launch.py \
    --config-name saes.gdino.coco \
    experiment.use_wandb=False \
    num_gpus=$WORLD_SIZE \
    runner.name="INTERPRETATION" \
    runner.mode="INFER" \
    runner.logs_period=20 \
    model.checkpoint_path="./artifacts/topk_fast_32/model_8000.pth" \
    dataset.infer=COCO_MINI \
    feature.path=$FEATURE_DIR \
    sae.variant=$SAE_VARIANT \
    feature.batch_size=$BATCH_SIZE \
    experiment.name=$EXPERIMENT_NAME \
    experiment.wandb_project_name="BlueGlass" \
    experiment.wandb_entity_name="intellabs" \
    experiment.output_dir="./experiment/${EXPERIMENT_NAME}"
    
