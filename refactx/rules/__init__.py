"""Rules for ReFactX: constrained Vadalog rule generation and the full mode.

* :mod:`vadalog_constraint` -- character-level constraint for one Vadalog rule
* :mod:`token_mask`         -- characters -> allowed tokens (sorted-vocab walk)
* :mod:`vadalog_kg`         -- a KG reasoned by vadalog-portable
* :mod:`generation`         -- ``<rule>`` pattern (RuleGeneration)
* :mod:`full_mode`          -- free generation with ``<kg>`` and ``<rule>``
"""
from refactx.rules.vadalog_constraint import VadalogRuleConstraint, build_trie  # noqa: F401
from refactx.rules.token_mask import SortedVocab  # noqa: F401


def __getattr__(name):  # lazy: these need vadalog-portable / torch
    if name in ("VadalogKG", "RuleResult"):
        from refactx.rules import vadalog_kg
        return getattr(vadalog_kg, name)
    if name == "RuleGeneration":
        from refactx.rules.generation import RuleGeneration
        return RuleGeneration
    if name in ("FullModeSession", "build_full_mode_processor", "Trace", "segments"):
        from refactx.rules import full_mode
        return getattr(full_mode, name)
    raise AttributeError(name)
