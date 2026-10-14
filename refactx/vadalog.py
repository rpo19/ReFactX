"""``<vadalog>{code}</vadalog>`` — constrained Vadalog generation.

Lets the model emit Vadalog rules inside a ``<vadalog> ... </vadalog>`` command,
collects the code, hands it to the Vadalog engine and stores the result on the
generation state so the rest of the answer can use it.

* Phase 0 — plumbing: collect the block, execute it, store the result.
* Phase 1 — **grammar-constrained** decoding: an xgrammar-compiled Vadalog
  grammar restricts every token so the block is always syntactically valid
  (Challenge 1).  Set ``grammar=False`` to disable.

Entity/relation/variable grounding (Challenge 2) is layered on in later phases —
see ``VadalogPlan.md``.
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
                 engine=None, execute=True, max_code_tokens=None,
                 grammar=True, grammar_ebnf=None, lexicon=None, entities=None):
        super().__init__(state, tokenizer, start_idx)
        self.open_text = open_text
        self.close_text = close_text
        self.engine = engine if engine is not None else VadalogEngine()
        self.execute = execute
        self.max_code_tokens = max_code_tokens
        self._close_ids = self._encode(close_text)
        self._forced = []
        self._pending_code = None
        # Grammar constraint (Challenge 1).  Falls back to unconstrained when
        # xgrammar is unavailable or compilation fails.
        self._grammar = None
        self._matcher = None
        self._bitmask = None
        self._fed = 0
        self.grammar_error = None
        if grammar:
            try:
                from refactx.vadalog_grammar import VadalogGrammar
                ebnf = grammar_ebnf
                if lexicon:
                    from refactx.vadalog_lexicon import build_grounded_ebnf
                    ebnf = build_grounded_ebnf(open_text, close_text, **lexicon)
                self._grammar = VadalogGrammar(tokenizer, open_text, close_text, ebnf)
                self._matcher = self._grammar.new_matcher()
                self._bitmask = self._grammar.allocate_bitmask()
            except Exception as exc:  # pragma: no cover - environment dependent
                self._grammar = None
                self.grammar_error = str(exc)
        # Dynamic entity grounding for large entity sets (Phase 2b).
        self._entities = None
        self.entity_error = None
        if entities:
            try:
                from refactx.vadalog_entities import EntityGrounder
                self._entities = EntityGrounder(tokenizer, entities)
            except Exception as exc:  # pragma: no cover - environment dependent
                self._entities = None
                self.entity_error = str(exc)
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

        # A forced closing tag is being emitted (safety net; bypasses grammar).
        if self._forced:
            token = self._forced.pop(0)
            mask[mask_idx, :] = -math.inf
            if 0 <= token < mask.shape[-1]:
                mask[mask_idx, token] = 0
            if not self._forced:
                self._finish(self._pending_code)
            return

        # The model (or the grammar) closed the block.
        if self.close_text in text:
            mask[mask_idx, :] = 0
            self._finish(text.split(self.close_text, 1)[0])
            return

        # Too long without closing: force the closing tag.
        if (self.max_code_tokens is not None
                and len(sequence) >= self.max_code_tokens):
            self._forced = list(self._close_ids)
            self._pending_code = text  # code at the trigger, close excluded
            token = self._forced.pop(0)
            mask[mask_idx, :] = -math.inf
            if 0 <= token < mask.shape[-1]:
                mask[mask_idx, token] = 0
            if not self._forced:
                self._finish(self._pending_code)
            return

        if self._grammar is not None:
            # Advance the grammar matcher with any tokens not fed yet.
            for token in sequence[self._fed:]:
                if not self._matcher.accept_token(token):
                    # Mask and matcher went out of sync; give up constraining.
                    self._grammar = None
                    self.grammar_error = "matcher rejected a generated token"
                    break
            self._fed = len(sequence)
            if self._grammar is not None:
                if self._matcher.is_completed():
                    mask[mask_idx, :] = 0
                    self._finish(text.split(self.close_text, 1)[0])
                    return
                self._grammar.apply_mask(mask, mask_idx, self._matcher, self._bitmask)
                if self._entities is not None:
                    self._entities.restrict(mask, mask_idx, text)
                return

        mask[mask_idx, :] = 0  # unconstrained inside the block
        if self._entities is not None:
            self._entities.restrict(mask, mask_idx, text)
