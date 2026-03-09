"""
Adapter to make RefactX usable as an Outlines backend.
This module provides a lightweight adapter that exposes a logits-processor-like
class compatible with Outlines' OutlinesLogitsProcessor interface.

Notes / TODOs:
- This is an initial scaffold. The real integration points depend on how
  RefactX exposes logits and beam search internals; for now we provide a
  stateless processor that can be expanded later.
- We intentionally avoid beam-search logic per instructions.
"""

try:
    from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
except Exception:
    # Outlines may not be installed in every environment. Provide a compatible
    # fallback base class so this module can be imported during local development.
    class OutlinesLogitsProcessor:
        def __init__(self, tensor_library_name: str = "torch"):
            self.tensor_library_name = tensor_library_name

        def reset(self):
            pass

        def process_logits(self, input_ids, logits):
            """Return processed logits. Override in subclass."""
            return logits


class RefactXLogitsProcessor(OutlinesLogitsProcessor):
    """Simple adapter that uses RefactX scoring/masking logic to adjust logits.

    Currently a scaffold: it exposes the OutlinesLogitsProcessor API and
    provides hooks (methods) where RefactX internals can be called.
    """

    def __init__(self, refactx_index=None, tensor_library_name: str = "torch"):
        super().__init__(tensor_library_name=tensor_library_name)
        self.index = refactx_index

    def reset(self):
        # Reset any state before a new generation
        pass

    def process_logits(self, input_ids, logits):
        # TODO: call into RefactX to mask/adjust logits based on `self.index`
        # For now, return logits unmodified.
        return logits


# Helper to create a backend-compatible factory (optional)

def get_refactx_logits_processor(refactx_index=None, tensor_library_name: str = "torch"):
    return RefactXLogitsProcessor(refactx_index=refactx_index, tensor_library_name=tensor_library_name)
