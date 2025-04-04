# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

import copy
from collections import defaultdict
import torch
from blueglass.modeling.saes import AutoEncoder
from typing import Dict, Any, Callable, Optional, Union


def intercept_manager() -> "InterceptManager":
    return _INTERCEPT_MANAGER


class Patcher:
    def __init__(self, pattern_name: str, patch_fn: Callable[..., Any]):
        self.pattern_name = pattern_name
        self.patch_fn = patch_fn

    def patch(self, *args, **kwargs) -> Any:
        return self.patch_fn(*args, **kwargs)


class StubPatcher(Patcher):
    def __init__(self):
        super().__init__("stub", lambda: 0)

    def patch(self, *args, **kwargs):
        assert isinstance(
            args[1], torch.Tensor
        ), "args[1] is not a tensor and returning it fail the pipeline. Please fix this if you are using this as a dummy patcher"
        return args[1]


class StandardPatcher(Patcher):
    """
    A standard patcher used when a function, rather than an nn.Module, is provided to transform inputs at an intermediate layer level.
    """

    def patch(self, *args, **kwargs) -> Any:
        process_name = kwargs["process_name"]
        target = kwargs["target"]
        if self.patch_fn is None:
            return target

        if self.patch_fn:
            return self.patch_fn(*args, **kwargs)
        return intercept_manager().patcher(self.pattern_name).patch(*args, **kwargs)


class SAEPatcher(Patcher):
    """
    A patcher designed for use when an nn.Module is passed to modify inputs of specific intermediate layers.
    """

    def __init__(self, name: str, sae: AutoEncoder):
        self.name = name
        self.sae = sae

    def patch(self, *args, **kwargs) -> Any:
        name = args[0]
        sae_input = args[1]
        sae_input_flat = sae_input.view(-1, sae_input.shape[-1])
        assert self.name == name, "sae name passed and registered name are not the same"

        sae_input_dict = {"features": sae_input_flat}
        sae_output_flat = self.sae(sae_input_dict)["pred_features"]
        sae_output = sae_output_flat.view(sae_input.shape)
        return sae_output


class Recorder:
    """
    Records features and structures them in the standard format.

    records: [{subpattern_name: Tensor}]

    """

    def __init__(self, name: str):
        self.name = name
        self.records = defaultdict(list)

    def process(self, items: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(items)

    def fetch_records(self) -> Dict[str, Any]:
        raise NotImplementedError("recorder has no records.")

    def record(self, name: str, items: Dict[str, Any]):
        raise NotImplementedError("override in child class.")

    def reset(self):
        self.records.clear()


class StubRecorder(Recorder):
    def __init__(self):
        super().__init__("stub")

    def process(self, items: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Unsupported in stub recorder.")

    def fetch_records(self) -> Dict[str, Any]:
        raise NotImplementedError("Unsupported in stub recorder.")

    def record(self, name: str, items: Dict[str, Any]):
        pass


class StandardRecorder(Recorder):
    def record(self, name: str, items: Dict[str, Any]):
        self.records[name].append(self.process(items))

    def fetch_records(self) -> Dict[str, Any]:
        return dict(self.records)


class InterceptManager:
    def __init__(self):
        self._registered_recorders: Dict[str, Recorder] = {}
        self._registered_patchers: Dict[str, Patcher] = {}
        self._stub_recorder_inst = StubRecorder()
        self._stub_patcher_inst = StubPatcher()

    def register(
        self,
        name: str,
        recorder: Optional[Recorder] = None,
        patcher: Optional[Patcher] = None,
    ):
        if recorder is not None:
            assert name not in self._registered_recorders, "name already registered."
            self._registered_recorders[name] = recorder

        if patcher is not None:
            assert name not in self._registered_patchers, "name already registered."
            self._registered_patchers[name] = patcher

    def deregister(
        self, name: str, record: bool = False, patch: bool = False
    ) -> Optional[Union[Recorder, Patcher]]:
        if record:
            assert name in self._registered_recorders, "name not registered."
            return self._registered_recorders.pop(name)

        if patch:
            assert name in self._registered_patchers, "name not registered."
            return self._registered_patchers.pop(name)

    def recorder(self, name: str):
        return self._registered_recorders.get(name, self._stub_recorder_inst)

    def patcher(self, name: str):
        return self._registered_patchers.get(name, self._stub_patcher_inst)

    def __repr__(self):
        return "InterceptManager"


_INTERCEPT_MANAGER = InterceptManager()
