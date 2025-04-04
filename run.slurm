#!/bin/bash
#SBATCH --partition=g24                     # Specify the partition
#SBATCH --gres=gpu:7                        # Request GPUs
#SBATCH --cpus-per-task=80                  # Request CPU cores
#SBATCH --nodelist=isl-gpu29                # Specify the node
#SBATCH --job-name=extract                  # Set job name
#SBATCH --mail-user=syed.qutub@intel.com    # Email for notifications
#SBATCH --mail-type=END                     # Send email when job ends
#SBATCH --output=job.%J.out                 # Standard output file
#SBATCH --error=job.%J.err                  # Standard error file

# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

#!/usr/bin/bash

# -------------------------------------
# Setup Variables Here.
# -------------------------------------

export TOKENIZERS_PARALLELISM=false
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
export CUDA_VISIBLE_DEVICES=0,1
export DATASET_DIR="/nwstore/datasets"
export WEIGHTS_DIR="/nwstore/weights"
# export FEATURE_DIR="/nwstore/harshal/blue_lens"
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
micromamba activate xai

echo "Using PYTHON : $(which python)"
nvidia-smi

# batch_size=131072
python launch.py \
    --config-name saes.gdino.coco \
    experiment.use_wandb=True \
    num_gpus=$WORLD_SIZE \
    dataset.infer=COCO_MINI \
    dataset.train=COCO_TRAIN \
    dataset.test=COCO_MINI \
    feature.path=$FEATURE_DIR \
    sae.variant=TOPK_FAST \
    feature.batch_size=65536 \
    experiment.name=sae.gdino.coco_train_E32
    

# sbatch -p g24 --gres=gpu:7 -c 80 --nodelist=isl-gpu8 -J extract --mail-user=syed.qutub@intel.com --mail-type=END run.sh
# nohup ./run.sh > training.log 2>&1 &