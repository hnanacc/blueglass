# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import os
import platform
import resource
import cv2
import numpy as np
import wandb
import torch
import torch.multiprocessing as mp

from blueglass.third_party.detectron2.utils import logger
from blueglass.configs import BLUEGLASSConf, to_dict


def setup_wandb(conf: BLUEGLASSConf):
    wandb.init(
        project=conf.experiment.wandb_project_name,
        entity=conf.experiment.wandb_entity_name,
        config=to_dict(conf),
        name=conf.experiment.name,
        group=f"DDP-{conf.experiment.name}" if conf.num_gpus > 0 else None,
    )


def default_setup(conf: BLUEGLASSConf):
    assert platform.system() == "Linux", "unsupported platform"

    # setup multiprocess runs.
    mp.set_sharing_strategy("file_system")

    if platform.system() != "Windows":
        require_method = "fork"
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != require_method:
            print(
                f"Multi-processing start method `{require_method}` is "
                f"different from the previous setting `{current_method}`."
                f"It will be force set to `{require_method}`. You can change "
                f"this behavior by changing `mp_start_method` in your config."
            )
        mp.set_start_method(require_method, force=True)

    # disable opencv threads.
    # overloads systems.
    cv2.setNumThreads(0)

    workers_per_gpu = max(2, conf.num_cpus - 1)

    # setup OMP threads to prevent overload.
    if "OMP_NUM_THREADS" not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        print(f"Setting OMP_NUM_THREADS={omp_num_threads}.")
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # setup MKL threads to prevent overload.
    if "MKL_NUM_THREADS" not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        print(f"Setting MKL_NUM_THREADS={mkl_num_threads}.")
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)

    # increase no. allowed file descriptors.
    softlim, hardlim = resource.getrlimit(resource.RLIMIT_NOFILE)
    softlim = min(max(4096, softlim), hardlim)
    resource.setrlimit(resource.RLIMIT_NOFILE, (softlim, hardlim))

    # limit the no. of cpus so other
    # processes can share. ex. multi-gpu
    torch.set_num_threads(conf.num_cpus)
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(conf.experiment.seed)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(conf.experiment.seed)

    logger.setup_logger(conf.experiment.output_dir, name="detectron2")
    logger.setup_logger(conf.experiment.output_dir, name="fvcore")
    logger.setup_logger(conf.experiment.output_dir, name="blueglass")

    if conf.experiment.use_wandb:
        setup_wandb(conf)


def default_teardown(conf: BLUEGLASSConf):
    if conf.experiment.use_wandb:
        wandb.finish()
