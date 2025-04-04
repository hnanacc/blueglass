# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import logging
from detectron2.utils import comm


class RankFormatter(logging.Formatter):
    def format(self, record):
        rank = getattr(comm, "get_rank", lambda: 0)()
        record.rank = rank
        return super().format(record)


def setup_blueglass_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Avoid duplicate handlers on repeated calls

    handler = logging.StreamHandler()
    formatter = RankFormatter(
        "[Rank %(rank)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
