from ast import For
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional


from sglang.srt.utils.ubatch_utils import UBatchSlices

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class ForwardContext:
    def __init__(
        self,
        forward_batch: Optional["ForwardBatch"] = None,
        attention_layers: Optional[List[Any]] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
    ):
        self.forward_batch = forward_batch
        self.attention_layers = attention_layers
        self.ubatch_slices = ubatch_slices
        self.ubatch_contexts: Optional[List["ForwardContext"]] = None
        
        if self.ubatch_slices is not None:
            # print(f"ForwardContext initialized with {len(self.ubatch_slices)} ubatch slices.") 
            ubatches = forward_batch.get_ubatches()
            self.ubatch_contexts = [
                ForwardContext(
                    forward_batch=ubatch,
                    attention_layers=attention_layers,
                    ubatch_slices=None,
                ) for ubatch in ubatches
            ]
            

    def set_forward_batch(self, forward_batch: "ForwardBatch"):
        self.forward_batch = forward_batch

    def set_attention_layers(self, layers: List[Any]):
        self.attention_layers = layers

    def set_ubatch_slices(self, ubatch_slices: UBatchSlices):
        self.ubatch_slices = ubatch_slices
        


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        return None
    return _forward_context

def replace_forward_context(new_context: ForwardContext):
    global _forward_context
    _forward_context = new_context
    # print(f"Forward context replaced. batch size: {new_context.forward_batch.batch_size}")


@contextmanager
def set_forward_context(
    forward_batch: "ForwardBatch",
    attention_layers: List[Any],
    ubatch_slices: Optional[UBatchSlices] = None,
):
    global _forward_context
    prev_forward_context = _forward_context
    _forward_context = ForwardContext(forward_batch, attention_layers, ubatch_slices)
    
    print(f"set_forward_context: {forward_batch.batch_size=}, {len(attention_layers)=}, {ubatch_slices=}")
    try:
        yield
    finally:
        _forward_context = prev_forward_context
