# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from blueglass.utils.logger_utils import setup_blueglass_logger
import copy
import pandas as pd
from torch import nn
from torch.utils.hooks import RemovableHandle
from blueglass.configs import BLUEGLASSConf, InterceptMode, FeaturePattern
from .processors.build import build_records_processor
from .hooks import build_hooknames_from_names
from .accessors import (
    Recorder,
    Patcher,
    StandardRecorder,
    intercept_manager,
)

from typing import List, Tuple, Dict, Union, Optional, Any

logger = setup_blueglass_logger(__name__)


class InterceptHook:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, module: nn.Module, inputs: Any, outputs: Any):
        intercept_manager().recorder(self.name).record(
            module.name, {"inputs": inputs, "outputs": outputs}
        )
        return (
            intercept_manager()
            .patcher(self.name)
            .patch(module.name, {"inputs": inputs, "outputs": outputs})
        )


class HookedSubInterceptor:
    def __init__(
        self,
        conf: BLUEGLASSConf,
        sub_model: nn.Module,
        recorders_per_name: Optional[Dict[str, Recorder]] = None,
        patchers_per_name: Optional[Dict[str, Patcher]] = None,
    ):
        self.sub_model = sub_model
        self.recorders_per_name = recorders_per_name
        self.patchers_per_name = patchers_per_name

        self.used_names = []

        if recorders_per_name:
            self.used_names.extend(list(recorders_per_name.keys()))

        if patchers_per_name:
            self.used_names.extend(list(patchers_per_name.keys()))

        self.hooknames_per_name = build_hooknames_from_names(
            conf, self.sub_model, self.used_names
        )
        self.attach_hook_metadata()

    def attach_hook_metadata(self):
        for hooknames in self.hooknames_per_name.values():
            for name in hooknames:
                setattr(self.sub_model.get_submodule(name), "name", name)

    def pre_forward(self, record: bool, patch: bool):
        ctx, handles_per_hookname = {}, {}

        if record:
            assert (
                self.recorders_per_name is not None
            ), "Expected recorders when record is enabled."
            for name, recorder in self.recorders_per_name.items():
                intercept_manager().register(name, recorder=recorder)
                handles_per_hookname.update(
                    {
                        hname: self.sub_model.get_submodule(
                            hname
                        ).register_forward_hook(InterceptHook(name))
                        for hname in self.hooknames_per_name[name]
                    }
                )

        if patch:
            assert (
                self.patchers_per_name is not None
            ), "Expected patchers when patch is enabled."
            for name, patcher in self.patchers_per_name.items():
                intercept_manager().register(name, patcher=patcher)
                handles_per_hookname.update(
                    {
                        hname: self.sub_model.get_submodule(
                            hname
                        ).register_forward_hook(InterceptHook(name))
                        for hname in self.hooknames_per_name[name]
                    }
                )

        ctx["handles_per_hookname"] = handles_per_hookname

        return ctx

    def pos_forward(
        self,
        record: bool,
        patch: bool,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:

        if record or patch:
            assert (
                "handles_per_hookname" in ctx
            ), "Expected handles from hooks when record or patch is enabled."
            assert isinstance(
                ctx["handles_per_hookname"], Dict
            ), "Expected handles_per_hookname to be a Dict."

            for hndl in ctx["handles_per_hookname"].values():
                assert isinstance(
                    hndl, RemovableHandle
                ), "Expected handle to be RemovableHandle."
                hndl.remove()

        if record:
            assert (
                self.recorders_per_name is not None
            ), "Expected recorders when record is enabled."
            ctx["records_per_name"] = {
                name: intercept_manager().deregister(name, record=True)
                for name in self.recorders_per_name
            }

        if patch:
            assert (
                self.patchers_per_name is not None
            ), "Expected patchers when patch is enabled."
            for name in self.patchers_per_name:
                intercept_manager().deregister(name, patch=True)

        return ctx


class ManualSubInterceptor:
    def __init__(
        self,
        conf: BLUEGLASSConf,
        recorders_per_name: Optional[Dict[str, Recorder]] = None,
        patchers_per_name: Optional[Dict[str, Patcher]] = None,
    ):
        self.conf = conf
        self.recorders_per_name = recorders_per_name
        self.patchers_per_name = patchers_per_name

    def pre_forward(self, record: bool, patch: bool) -> Dict[str, Any]:
        if record:
            assert (
                self.recorders_per_name is not None
            ), "Expected recorders when record is enabled."
            for name, recorder in self.recorders_per_name.items():
                intercept_manager().register(name, recorder=recorder)

        if patch:
            assert (
                self.patchers_per_name is not None
            ), "Expected patchers when patch is enabled."
            for name, patcher in self.patchers_per_name.items():
                intercept_manager().register(name, patcher=patcher)

        return {}

    def pos_forward(
        self, record: bool, patch: bool, ctx: Dict[str, Any]
    ) -> Dict[str, Any]:

        if record:
            assert (
                self.recorders_per_name is not None
            ), "Expected recorder when record is enabled."
            ctx["records_per_name"] = copy.deepcopy(
                {
                    name: intercept_manager().deregister(name, record=True)
                    for name in self.recorders_per_name
                }
            )
            clear_records = [
                self.recorders_per_name[name].reset()
                for name in self.recorders_per_name
            ]

        if patch:
            assert (
                self.patchers_per_name is not None
            ), "Expected patcher when patch is enabled."
            for name in self.patchers_per_name:
                intercept_manager().deregister(name, patch=True)

        return ctx


class FeatureInterceptor(nn.Module):
    def __init__(
        self,
        conf: BLUEGLASSConf,
        model: nn.Module,
        recorders_per_name: Optional[Dict[str, Recorder]] = None,
        patchers_per_name: Optional[Dict[str, Patcher]] = None,
    ):
        super().__init__()
        self.conf = conf
        self.sub_model = model
        self.recorders_per_name = recorders_per_name
        self.patchers_per_name = patchers_per_name

        self.sub_incpt = self.build_sub_interceptor()
        self.rec_procr = build_records_processor(conf)

    def build_sub_interceptor(self):
        match self.conf.feature.intercept_mode:
            case InterceptMode.HOOKS:
                return HookedSubInterceptor(
                    self.conf,
                    self.sub_model,
                    self.recorders_per_name,
                    self.patchers_per_name,
                )
            case InterceptMode.MANUAL:
                return ManualSubInterceptor(
                    self.conf, self.recorders_per_name, self.patchers_per_name
                )
            case unsupported:
                raise ValueError(f"unsupported subinterceptor: {unsupported}")

    def forward(
        self,
        *args,
        record: bool = False,
        patch: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]]:
        context = self.sub_incpt.pre_forward(record, patch)
        outputs = self.sub_model(*args, **kwargs)
        context = self.sub_incpt.pos_forward(record, patch, context)

        if record:
            assert (
                "records_per_name" in context
            ), "Expected records in context when record is enabled."
            return outputs, self.rec_procr(context["records_per_name"])

        return outputs
