# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from typing import Callable, List, Dict, Any, Union
from blueglass.configs import BLUEGLASSConf, Model, FeaturePattern
from ..accessors import Recorder
from ..types import DistFormat

from .gdino import GDINOAligner
from .dino_detr import DINOAligner
from .detr import DETRAligner
from .genu import GENUAligner
from .florence import FlorenceAligner

RecordedFormatType = Union[Dict[FeaturePattern, Recorder], Dict[str, Recorder]]
SchemaedFormatType = Dict[str, DistFormat]


def compute_confusion_from_unprocessed(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    return features


def _prepare_extender_processor(
    conf: BLUEGLASSConf,
) -> List[Callable[[RecordedFormatType], RecordedFormatType]]:
    procs = []

    if conf.feature.compute_confusion:
        procs.append(compute_confusion_from_unprocessed)

    return procs


def _prepare_sequence_processor(
    conf: BLUEGLASSConf,
) -> Callable[[RecordedFormatType], SchemaedFormatType]:
    return {
        Model.GDINO: GDINOAligner,
        Model.GENU: GENUAligner,
        Model.DINO_DETR: DINOAligner,
        Model.DETR: DETRAligner,
        Model.FLORENCE: FlorenceAligner,
    }[conf.model.name](conf)


class ComposedAligner:
    def __init__(
        self,
        recorder_processor: Callable[[RecordedFormatType], SchemaedFormatType],
        extender_processor: List[Callable[[RecordedFormatType], RecordedFormatType]],
    ):
        self.rec_proc = recorder_processor
        self.ext_proc = extender_processor

    def __call__(self, recorder_per_pattern: RecordedFormatType) -> SchemaedFormatType:
        for procfn in self.ext_proc:
            parsed = procfn(parsed)

        return self.rec_proc(recorder_per_pattern)


def build_records_aligners(conf: BLUEGLASSConf):
    rec_proc = _prepare_sequence_processor(conf)
    ext_proc = _prepare_extender_processor(conf)

    return ComposedAligner(rec_proc, ext_proc)
