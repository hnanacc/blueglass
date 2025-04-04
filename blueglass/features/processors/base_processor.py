# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
import torch
from torch import Tensor
import time
from blueglass.configs import BLUEGLASSConf, FeaturePattern
from ..types import IOFrame, DistFormat
from ..schema import MinimalSchemaFrame
from ..accessors import Recorder

from typing import Callable, List, Dict, Union, Tuple


def process_feature(
    infer_id, total_infer_ids, subpattern, feature, std_io_processed, mapper
):
    """
    Process a single feature using the mapper function.
    """
    start = time.perf_counter()
    result = mapper(infer_id, total_infer_ids, subpattern, feature, std_io_processed)
    end = time.perf_counter()
    return result


class Processor:
    def __init__(self, conf: BLUEGLASSConf):
        self.conf = conf
        self.cpu_count = conf.num_cpus - 2
        self.use_multiprocessing = False

    def __call__(
        self, recorders_per_name: Dict[str, Recorder]
    ) -> Dict[str, DistFormat]:
        """
        Calls the appropriate function based on multiprocessing flag.
        """
        start_time = time.perf_counter()
        result = self.single_threaded_execution(recorders_per_name)
        end_time = time.perf_counter()
        return result

    def single_threaded_execution(
        self, recorder_per_pattern: Dict[str, Recorder]
    ) -> Dict[str, DistFormat]:
        std_io = self.process_io(recorder_per_pattern.pop(FeaturePattern.IO))
        result = {
            schema_name: schema_item
            for pattern, records_per_subpattern in recorder_per_pattern.items()
            for schema_name, schema_item in self.to_schema(
                pattern,
                self.process_per_pattern(pattern, records_per_subpattern, std_io),
            ).items()
        }

        return result

    def _norm(self, name: str):
        return name.replace(".", "_")

    def _schema_name(self, layer_id: int, pattern: str, subpattern: str) -> str:
        return f"layer_{layer_id}.{self._norm(pattern)}.{self._norm(subpattern)}"

    def to_schema_mp(
        self,
        pattern: str,
        records_per_subpattern: Dict[Tuple[int, str], List[MinimalSchemaFrame]],
    ) -> Dict[str, DistFormat]:
        ## todo: multiprocess this and evaluate it for performance benefits
        start_time = time.perf_counter()

        schema_results = {}

        # Use ThreadPoolExecutor to parallelize the outer loop
        diskfmt_func = self._convert_to_dist_format
        schem_func = self._schema_name
        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = {
                executor.submit(
                    process_subpattern,
                    layer_id,
                    pattern,
                    subpattern,
                    records,
                    diskfmt_func,
                    schem_func,
                ): (layer_id, subpattern)
                for (layer_id, subpattern), records in records_per_subpattern.items()
            }

            # Collect results as they are completed
            for future in as_completed(futures):
                key, result = future.result()
                schema_results[key] = result

        end_time = time.perf_counter()
        return schema_results

    def to_schema(
        self,
        pattern: str,
        records_per_subpattern: Dict[Tuple[int, str], List[MinimalSchemaFrame]],
    ) -> Dict[str, DistFormat]:
        schema_results = {}
        start_time = time.perf_counter()
        for (layer_id, subpattern), records in records_per_subpattern.items():
            key = self._schema_name(layer_id, pattern, subpattern)
            """
            len(records) is the number of inferneces for example contrastive models can only have one infererence hence infer_id = 0 
            but generative models can have n inferneces dependending on token generation and end of sequence hence it can have infere_id = [0, 1, ...., n]
            """
            # schema_results[self._schema_name(layer_id, pattern, subpattern)] = pd.concat([self._convert_to_dist_format(convert_tensors_to_lists(record)) for record in records], axis=0, ignore_index=True)
            schema_results[self._schema_name(layer_id, pattern, subpattern)] = (
                pd.concat(
                    [self._convert_to_dist_format(record) for record in records],
                    axis=0,
                    ignore_index=True,
                )
            )
        end_time = time.perf_counter()
        return schema_results

    def process_per_pattern(
        self, pattern: Union[FeaturePattern, str], recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        """
        Returns the standardized record: Dict[item_name, item_part_frame]
        where  item_name is model_dataset_pattern_subpattern: FeatureFrame (in schema).
        """
        match pattern:
            case (
                FeaturePattern.DET_DECODER_RESID_MLP
                | FeaturePattern.DET_DECODER_RESID_MHA
            ):
                result = self.process_det_decoder_resid(recorder, std_io)
            case FeaturePattern.DET_DECODER_MHA:
                result = self.process_det_decoder_mha(recorder, std_io)
            case FeaturePattern.DET_DECODER_MLP:
                result = self.process_det_decoder_mlp(recorder, std_io)
            case (
                FeaturePattern.LLM_DECODER_RESID_MHA
                | FeaturePattern.LLM_DECODER_RESID_MLP
            ):
                result = self.process_llm_decoder_resid(recorder, std_io)
            case FeaturePattern.LLM_DECODER_MHA:
                result = self.process_llm_decoder_mha(recorder, std_io)
            case FeaturePattern.LLM_DECODER_MLP:
                result = self.process_llm_decoder_mlp(recorder, std_io)
            case unsupported:
                raise NotImplementedError(
                    f"unsupported feature pattern: {unsupported}."
                )

        return result

    def _decompose_with(
        self, mapper: Callable, recorder: Recorder, std_io_processed: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        """
        Inside the above function:
        # std_io_processed = self.build_std_io_dict(std_io_processed, recorder)
        # batch_size = len(std_io_dict[list(std_io_dict.keys())[0]])
        # records_len = len(record["image_id"])
        """
        fetched_records = recorder.fetch_records()

        returns = defaultdict(list)
        process_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for layer_name, layer_recs in fetched_records.items():
                layer_id = self._infer_layer_id_from_name(layer_name)
                total_infer_ids = len(layer_recs)
                for infer_id, infer_recs in enumerate(layer_recs):
                    assert isinstance(infer_recs, dict), "unexpected infer_recs."
                    for subpattern, feature in infer_recs.items():
                        assert isinstance(feature, torch.Tensor), "unexpected feature."
                        # Submit the task to the executor
                        feature_copy = feature.clone()
                        future = executor.submit(
                            process_feature,
                            infer_id,
                            total_infer_ids,
                            subpattern,
                            feature_copy,
                            std_io_processed,
                            mapper,
                        )
                        futures.append(((layer_id, subpattern), future))

            # Collect results as they complete
            for (layer_id, subpattern), future in futures:
                try:
                    result = future.result()
                    returns[(layer_id, subpattern)].append(result)
                except Exception as e:
                    print(
                        f"Error processing feature for layer {layer_id}, subpattern {subpattern}: {e}"
                    )

        process_end = time.perf_counter()
        return returns

    def _decompose_with_old(
        self, mapper: Callable, recorder: Recorder, std_io_processed: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:

        fetched_records = recorder.fetch_records()
        process_start = time.perf_counter()

        returns = defaultdict(list)
        for layer_name, layer_recs in fetched_records.items():
            layer_id = self._infer_layer_id_from_name(layer_name)
            total_infer_ids = len(layer_recs)
            for infer_id, infer_recs in enumerate(layer_recs):
                assert isinstance(infer_recs, Dict), "unexpected infer_recs."
                for subpattern, feature in infer_recs.items():
                    assert isinstance(feature, Tensor), "unexpected feature."
                    start = time.perf_counter()
                    returns[(layer_id, subpattern)].append(
                        mapper(
                            infer_id,
                            total_infer_ids,
                            subpattern,
                            feature,
                            std_io_processed,
                        )
                    )
                    end = time.perf_counter()
        process_end = time.perf_counter()
        return dict(returns)

    def _infer_layer_id_from_name(self, name: str):
        return int(name.split("_")[-1])

    def _convert_to_dist_format(self, records: List[Dict]) -> DistFormat:
        return pd.DataFrame(records)

    def process_io(self, recorder: Recorder) -> IOFrame:
        raise NotImplementedError("override in child.")

    def process_det_decoder_resid(
        self, recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        raise NotImplementedError("override in child.")

    def process_llm_decoder_resid(
        self, recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        raise NotImplementedError("override in child.")

    def process_det_decoder_mha(
        self, recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        raise NotImplementedError("override in child.")

    def process_det_decoder_mlp(
        self, recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        raise NotImplementedError("override in child.")

    def process_llm_decoder_mha(
        self, recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        raise NotImplementedError("override in child.")

    def process_llm_decoder_mlp(
        self, recorder: Recorder, std_io: IOFrame
    ) -> Dict[Tuple[int, str], List[Dict]]:
        raise NotImplementedError("override in child.")
