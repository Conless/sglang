from sglang.srt.utils.ubatch_utils import UBatchSlice, UBatchSlices
import torch
from typing import TYPE_CHECKING, Any, Optional
from schedflow.config import SchedFlowConfig
from schedflow.manager import SchedFlowManager


from schedflow.interface import OpSchedulerBase, SplitConfig
from schedflow.example.sglang.nanoflow import (
    NanoFlowScheduler,
    NanoFlowSchedulerConfig,
)
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


    
_manager = SchedFlowManager()
_scheduler: Optional[NanoFlowScheduler] = None


def get_scheduler(
    config: NanoFlowSchedulerConfig,
) -> NanoFlowScheduler:
    global _scheduler
    if isinstance(config, NanoFlowSchedulerConfig):
        _scheduler = NanoFlowScheduler(config)
    else:
        raise ValueError(f"Invalid scheduler config: {config}")
    return _scheduler

def get_manager(
    graph_module: torch.fx.GraphModule,
    config: SchedFlowConfig,
    scheduler: OpSchedulerBase,
    example_inputs: list[Any],
) -> SchedFlowManager:
    global _manager
    _manager.initialize(graph_module, config, scheduler, example_inputs)
    return _manager


def nano_ubatch_split(
    batch: "ForwardBatch",
    is_dummy_run: bool = False,
    use_cudagraph: bool = False,
) -> tuple[UBatchSlices, torch.Tensor]:
    """
    Prepare two UBatch-compatible nano-batch slices.

    - Uses nano_manager.prepare_nano_split to decide if splitting is beneficial
      (i.e., num_nano_batches > 1).
    - Computes a single token split point using custom logic to remain
      compatible with UBatch execution.
    """

    batch_size = batch.batch_size
    if batch.forward_mode.is_extend():
        num_tokens = batch.extend_seq_lens.tolist()
    elif batch.forward_mode.is_decode():
        num_tokens = [1] * batch_size
    else:
        raise ValueError(f"SchedFlow: unsupported forward mode: {batch.forward_mode}")
    # print(f"Preparing nano-batch split for {batch.input_ids.shape=} {batch_size=}, {num_tokens=}, {is_dummy_run=}, {use_cudagraph=}")
    split_config = _manager.prepare(
        batch_size,
        num_tokens=num_tokens,
        is_dryrun=is_dummy_run,
        use_cudagraph=use_cudagraph,
    )
    
    dp_size = 1
    total_num_tokens_across_dp = [sum(split_config.num_tokens_padded)]
    
    return (
        [
            UBatchSlice(
                request_slice=slice(
                    split_config.batch_indices[i],
                    split_config.batch_indices[i + 1],
                ),
                token_slice=slice(
                    split_config.split_indices[i],
                    split_config.split_indices[i + 1],
                ),
            )
            for i in range(split_config.num_nano_batches)
        ],
        torch.tensor(
            total_num_tokens_across_dp,
            device="cpu",
            dtype=torch.int32,
        )
    )
    
    