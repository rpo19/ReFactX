"""vLLM V1 logits processor for ReFactX constrained generation.

vLLM natively batches many requests and calls one logits processor for the whole
persistent batch.  This adapter keeps one independent
:class:`~refactx.generate.PatternConstrainedState` per request row and reuses the
same :class:`~refactx.generate.FactGeneration` /
:class:`~refactx.generate.CountBranchesGeneration` logic as the HuggingFace path.

Requests opt in with ``SamplingParams.extra_args={"refactx": True}``.  An
optional ``refactx_id`` lets callers retrieve the structured facts through
``pop_facts``.  If ``REFACTX_INDEX_TABLES`` is set, a per-request
``index_table`` extra arg selects the KB table through ``RoutedIndex``.

The custom logits-processor API is available in vLLM V1.  On pre-Ampere GPUs
use the Triton attention backend (see ``examples/vllm_offline.py``).

Environment::

    REFACTX_INDEX           index file / postgres:// / http(s):// (required)
    REFACTX_INDEX_TABLES    optional comma-separated KB table names
    REFACTX_DEFAULT_TABLE   default table when routing (default: first table)
    REFACTX_FACT_PATTERN    trigger (default: <fact>)
    REFACTX_COUNT_PATTERN   count trigger (default: <count>)
    REFACTX_EOT             fact terminator (default: " </fact>\\n")
    REFACTX_SENTINEL        1/true to enable exhausted-retrieval sentinel
    REFACTX_FORCE_CONSTRAINED
                            1/true to start every request in fact mode
                            (benchmarks/tests; skips trigger detection)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

# vLLM 0.25 imports ALLOWED_LAYER_TYPES, which older transformers versions did
# not export.  Keep this compatibility shim local to the optional vLLM path.
try:  # pragma: no cover - depends on the installed transformers version
    import transformers.configuration_utils as _tcu
    if not hasattr(_tcu, 'ALLOWED_LAYER_TYPES'):
        _tcu.ALLOWED_LAYER_TYPES = getattr(_tcu, 'ALLOWED_ATTENTION_LAYER_TYPES', ())
except Exception:
    pass

from vllm.v1.sample.logits_processor import (  # noqa: E402
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

import refactx  # noqa: E402
from refactx.generate import activate_constrained_fact, build_pattern_state  # noqa: E402
from refactx.index import RoutedIndex  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm import SamplingParams
    from vllm.config import VllmConfig

ENABLE_KEY = "refactx"
CORR_ID_KEY = "refactx_id"
INDEX_TABLE_KEY = "index_table"
INDEX_ENV = "REFACTX_INDEX"
TABLES_ENV = "REFACTX_INDEX_TABLES"


def _extra_args(params) -> dict:
    return getattr(params, "extra_args", None) or {}


def _enabled(params) -> bool:
    return bool(_extra_args(params).get(ENABLE_KEY))


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class _ReqState:
    """Per persistent-batch-row bookkeeping for one constrained request."""

    __slots__ = ("state", "out_ids", "consumed", "corr_id")

    def __init__(self, state, out_ids, corr_id):
        self.state = state
        self.out_ids = out_ids
        self.consumed = 0
        self.corr_id = corr_id


class RefactxLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        self.device = device
        model_cfg = vllm_config.model_config
        token_id = getattr(model_cfg, "tokenizer", None) or model_cfg.model
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            token_id,
            trust_remote_code=getattr(model_cfg, "trust_remote_code", False),
        )

        index_url = os.environ.get(INDEX_ENV)
        if not index_url:
            raise ValueError(
                f"{INDEX_ENV} is not set. Point it at the ReFactX index "
                "(a text file, postgresql:// URL, or http(s):// endpoint)."
            )
        self.index = self._load_index(index_url)

        self.fact_pattern = os.environ.get("REFACTX_FACT_PATTERN", "<fact>")
        self.count_pattern = os.environ.get("REFACTX_COUNT_PATTERN", "<count>")
        self.eot = os.environ.get("REFACTX_EOT", " </fact>\n")
        self.sentinel = _env_bool("REFACTX_SENTINEL", False)
        self.force_constrained = _env_bool("REFACTX_FORCE_CONSTRAINED", False)
        self._mask_buffer = None

        self._req: dict[int, _ReqState] = {}
        self._facts: dict[object, list[str]] = {}

    def _load_index(self, index_url):
        tables = os.environ.get(TABLES_ENV, "").strip()
        if not tables:
            if os.path.isfile(index_url):
                return refactx.load_index(index_url, tokenizer=self.tokenizer)
            return refactx.load_index(index_url)

        names = [name.strip() for name in tables.split(",") if name.strip()]
        if not names:
            return refactx.load_index(index_url)
        registry = {}
        for name in names:
            url = f"{index_url}?tablename={name}"
            registry[name] = refactx.load_index(url)
        default = os.environ.get("REFACTX_DEFAULT_TABLE", names[0])
        return RoutedIndex(registry, default=default)

    def _new_state(self, index_table=None):
        state = build_pattern_state(
            self.tokenizer,
            self.index,
            sentinel=self.sentinel,
            fact_pattern=self.fact_pattern,
            count_pattern=self.count_pattern,
            eot=self.eot,
        )
        state.index_table = index_table
        if self.force_constrained:
            activate_constrained_fact(
                state, self.index, sentinel=self.sentinel, eot=self.eot,
            )
        return state

    # -- vLLM LogitsProcessor interface ------------------------------------

    @classmethod
    def validate_params(cls, params: "SamplingParams") -> None:
        extra = _extra_args(params)
        if ENABLE_KEY in extra and not isinstance(extra[ENABLE_KEY], bool):
            raise ValueError(
                f"extra_args['{ENABLE_KEY}'] must be a bool "
                f"(got {type(extra[ENABLE_KEY]).__name__})."
            )
        if INDEX_TABLE_KEY in extra and not isinstance(extra[INDEX_TABLE_KEY], str):
            raise ValueError(
                f"extra_args['{INDEX_TABLE_KEY}'] must be a string."
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        if batch_update is None:
            return

        for index in batch_update.removed:
            self._finish(self._req.pop(index, None))

        for index, params, _prompt_ids, output_ids in batch_update.added:
            self._finish(self._req.pop(index, None))
            if _enabled(params):
                extra = _extra_args(params)
                self._req[index] = _ReqState(
                    self._new_state(extra.get(INDEX_TABLE_KEY)),
                    output_ids,
                    extra.get(CORR_ID_KEY),
                )

        for a_index, b_index, direct in batch_update.moved:
            a_entry = self._req.pop(a_index, None)
            b_entry = self._req.pop(b_index, None)
            if a_entry is not None:
                self._req[b_index] = a_entry
            if b_entry is not None and direct == MoveDirectionality.SWAP:
                self._req[a_index] = b_entry

    def _mask_row(self, logits):
        shape = (1, logits.shape[-1])
        buffer = self._mask_buffer
        if (
            buffer is None
            or buffer.shape != shape
            or buffer.dtype != logits.dtype
            or buffer.device != logits.device
        ):
            buffer = torch.empty(shape, dtype=logits.dtype, device=logits.device)
            self._mask_buffer = buffer
        return buffer

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._req:
            return logits

        n_rows = logits.shape[0]
        mask = None
        for i, req in self._req.items():
            # Drain all output tokens appended since the previous forward.  In
            # normal decode this is one token, but this also handles chunked
            # prefill and any multi-token append.
            out = req.out_ids
            if req.consumed < len(out):
                for token in out[req.consumed:]:
                    req.state.update(token)
                req.consumed = len(out)

            if i >= n_rows or not req.state.is_constrained():
                continue

            if mask is None:
                mask = self._mask_row(logits)
            mask.fill_(float("-inf"))
            sequence = list(req.state.token_ids)
            req.state.active_generation.constrain(sequence, mask, 0)
            # Preserve the model's relative scores for allowed tokens.
            logits[i].add_(mask[0])

        return logits

    # -- helpers / public extras -------------------------------------------

    def _finish(self, req: Optional[_ReqState]) -> None:
        if req is None or req.corr_id is None:
            return
        self._facts[req.corr_id] = [
            self.tokenizer.decode(triple_ids, skip_special_tokens=True)
            for triple_ids in req.state.generated_triples
        ]

    def pop_facts(self, corr_id) -> list[str]:
        """Return and clear the facts generated for ``corr_id``."""
        return self._facts.pop(corr_id, [])
