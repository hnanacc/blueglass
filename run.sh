# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

#!/usr/bin/bash

# -------------------------------------
# Setup Variables Here.
# -------------------------------------

export TOKENIZERS_PARALLELISM=false
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
export CUDA_VISIBLE_DEVICES=0
export DATASET_DIR="/nwstore/datasets"
export WEIGHTS_DIR="/nwstore/weights"
export FEATURE_DIR="/nwstore/blueglass/blue_lens_extract"


# -------------------------------------
# Setup Environment.
# -------------------------------------

# if ! command -v micromamba 2>&1 >/dev/null
# then
#     echo "No environment manager not found. Cannot proceed."
#     exit 1
# fi

echo "Found micromamba on the system."
eval "$(micromamba shell.hook bash)"
micromamba activate blueglass_env

echo "Using PYTHON : $(which python)"
nvidia-smi

# batch_size=131072
# BATCH_SIZE=16384
# BATCH_SIZE=32768
# BATCH_SIZE=49152
# BATCH_SIZE=57344
BATCH_SIZE=12
SAE_VARIANT=TOPK_FAST
DATASET="COCO"
EXPERIMENT_NAME="sae.gdino.${DATASET}_train_${SAE_VARIANT}_B${BATCH_SIZE}"
python launch.py \
    --config-name features.gdino.coco \
    experiment.use_wandb=False \
    num_gpus=$WORLD_SIZE \
    dataset.infer=COCO_MINI \
    dataset.train=COCO_MINI \
    dataset.test=COCO_MINI \
    feature.path=$FEATURE_DIR \
    sae.variant=$SAE_VARIANT \
    feature.batch_size=$BATCH_SIZE \
    experiment.name=$EXPERIMENT_NAME
    
