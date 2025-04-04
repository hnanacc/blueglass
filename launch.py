# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import warnings

warnings.warn = lambda *args, **kwargs: None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from blueglass.utils.launch import default_setup, default_teardown
from blueglass.data.datasets import register_custom_datasets
from blueglass.runners import build_runner
from blueglass.third_party.detectron2.engine import launch
from blueglass.configs import BLUEGLASSConf
import hydra


def process_main(conf: BLUEGLASSConf):
    default_setup(conf)
    register_custom_datasets(conf)

    if conf.runner.mode == "train":
        return build_runner(conf).train()
    if conf.runner.mode == "test":
        return build_runner(conf).test()
    if conf.runner.mode == "infer":
        return build_runner(conf).infer()

    raise ValueError(f"Unsupported run mode: {conf.runner.mode}.")


@hydra.main(version_base=None)
def main(conf: BLUEGLASSConf):
    try:
        launch(
            process_main,
            args=(conf,),
            dist_url="auto",
            machine_rank=conf.machine_rank,
            num_gpus_per_machine=conf.num_gpus,
            num_machines=conf.num_machines,
        )
    except KeyboardInterrupt:
        default_teardown(conf)


if __name__ == "__main__":
    main()
