"""``<vadalog>{code}</vadalog>`` — Phase 0 plumbing.

Lets the model emit Vadalog rules inside a ``<vadalog> ... </vadalog>`` command,
collects the code, hands it to the Vadalog engine and stores the result on the
generation state so the rest of the answer can use it.

Phase 0 intentionally only enforces the **closing tag**: generation inside the
block is otherwise unconstrained.  Grammar-constrained decoding (Challenge 1)
and entity/relation/variable lexicons (Challenge 2) are layered on in later
phases — see ``VadalogPlan.md``.
"""
from __future__ import annotations

import math
from copy import deepcopy

from transformers import ProcessorMixin

from refactx.generate import PatternConstrainedGeneration

DEFAULT_OPEN = "<vadalog>"
DEFAULT_CLOSE = "</vadalog>"


class VadalogEngine:
    """Best-effort adapter around the Vadalog engine.

    Accepts an explicit ``runner`` (``code: str -> result``) for tests/embedding;
    otherwise tries to import the ``vadalog`` package.  When the engine is
    unavailable, or a program fails, ``run`` returns a structured error instead
    of raising, so a bad rule cannot crash a run.
    """

    def __init__(self, runner=None):
        self._runner = runner
        self._load_error = None
        if self._runner is None:
            try:
                import vadalog as _vadalog
                if hasattr(_vadalog, "VadalogEngine"):
                    self._runner = _vadalog.VadalogEngine().run
                else:
                    self._runner = getattr(_vadalog, "run", None) \
                        or getattr(_vadalog, "execute", None)
            except Exception as exc:  # pragma: no cover - depends on install
                self._load_error = str(exc)

    @property
    def available(self):
        return self._runner is not None

    def run(self, code):
        """Execute ``code``; return ``{'ok', 'result', 'error'}``."""
        if self._runner is None:
            return {"ok": False, "result": None,
                    "error": f"vadalog engine not available: {self._load_error}"}
        try:
            return {"ok": True, "result": self._runner(code), "error": None}
        except Exception as exc:  # pragma: no cover - engine dependent
            return {"ok": False, "result": None, "error": str(exc)}


class VadalogGeneration(PatternConstrainedGeneration):
    """Collect a ``<vadalog> ... </vadalog>`` block and execute it.

    Phase 0 constraints: none — every token is allowed inside the block, so the
    model writes Vadalog freely.  The block ends at the literal closing tag; if
    ``max_code_tokens`` is exceeded the closing tag is forced so the block is
    always well-formed.  The (trimmed) code and the engine result are appended to
    ``state.vadalog_blocks``.
    """

    def __init__(self, state, tokenizer, start_idx,
                 open_text=DEFAULT_OPEN, close_text=DEFAULT_CLOSE,
                 engine=None, execute=True, max_code_tokens=None):
        super().__init__(state, tokenizer, start_idx)
        self.open_text = open_text
        self.close_text = close_text
        self.engine = engine if engine is not None else VadalogEngine()
        self.execute = execute
        self.max_code_tokens = max_code_tokens
        self._close_ids = self._encode(close_text)
        self._forced = []
        self._pending_code = None
        if not hasattr(state, "vadalog_blocks"):
            state.vadalog_blocks = []

    # -- tokenizer helpers (mirror KnowledgeGraphGeneration) ---------------
    def _tokenizer(self):
        return self.tokenizer.tokenizer if isinstance(self.tokenizer, ProcessorMixin) else self.tokenizer

    def _encode(self, text):
        return list(self._tokenizer().encode(text, add_special_tokens=False))

    def _decode(self, ids):
        return self._tokenizer().decode(ids)

    # -- completion --------------------------------------------------------
    def _finish(self, code_text):
        block = {"code": (code_text or "").strip(), "result": None}
        if self.execute:
            block["result"] = self.engine.run(block["code"])
        self.state.vadalog_blocks.append(block)
        self.done = True

    def constrain(self, sequence, mask, mask_idx):
        text = self._decode(sequence)

        # The model closed the block itself.
        if not self._forced and self.close_text in text:
            mask[mask_idx, :] = 0
            self._finish(text.split(self.close_text, 1)[0])
            return

        # Too long without closing: force the closing tag.
        if (not self._forced and self.max_code_tokens is not None
                and len(sequence) >= self.max_code_tokens):
            self._forced = list(self._close_ids)
            self._pending_code = text  # code as of the trigger, close excluded

        # Emit a queued closing-tag token.
        if self._forced:
            token = self._forced.pop(0)
            mask[mask_idx, :] = -math.inf
            if 0 <= token < mask.shape[-1]:
                mask[mask_idx, token] = 0
            if not self._forced:
                self._finish(self._pending_code)
            return

        mask[mask_idx, :] = 0  # unconstrained inside the block
