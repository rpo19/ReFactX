"""Full mode: free generation that can switch into ``<kg>`` or ``<rule>``.

The model writes free text.  Two trigger strings switch the logits processor
into a constrained mode for one block, after which generation is free again:

``<kg>``    one fact, ``<Subject> <relation> <Object> .``.  The subject comes
            from the entity index; the relations offered for it are
            ``kg.properties(subject)``: stored relations, relations derived
            by the Vadalog program (never stored: *unmaterialized* facts) and
            rules written earlier in the answer.  Objects come from
            ``kg.get``.
``<rule>``  one Vadalog rule, syntactically valid and grounded in the schema
            (see :mod:`refactx.rules.vadalog_constraint`); Vadalog evaluates
            it and the answer continues after a forced ``<result>`` block.

Everything is driven by one ``model.generate()`` call.
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field

from refactx.generate import (ConstrainedLogitsProcessor, ConstrainedStateList,
                              KnowledgeGraphGeneration)
from refactx.index import DictIndex
from refactx.rules.generation import RuleGeneration
from refactx.rules.token_mask import SortedVocab, raw_tokenizer

KG_OPEN, KG_CLOSE = "<kg>", " </kg>\n"
RULE_OPEN, RULE_CLOSE = "<rule>", " </rule>\n"


def fill_schema(messages, descriptions, placeholder="{SCHEMA}"):
    """Replace ``{SCHEMA}`` in the prompt with one line per predicate."""
    block = "\n".join("- " + d for d in descriptions)
    return [dict(m, content=m["content"].replace(placeholder, block)) for m in messages]


def build_entity_index(tokenizer, entities):
    """The ``<kg>`` subject index: ' <name>' for every entity."""
    tok = raw_tokenizer(tokenizer)
    index = DictIndex()
    index.set_tokenizer(tokenizer)
    for e in entities:
        index.add(tok.encode(" <%s>" % e, add_special_tokens=False))
    return index


def build_full_mode_processor(tokenizer, kg, entity_index=None, kg_mode=True,
                              rule_mode=True, num_beams=1, num_batches=1,
                              kg_kwargs=None, rule_kwargs=None):
    """A ConstrainedLogitsProcessor with the ``<kg>`` and ``<rule>`` patterns."""
    states = ConstrainedStateList('auto', num_beams=num_beams, num_batches=num_batches,
                                  debug_tokenizer=tokenizer)
    processor = ConstrainedLogitsProcessor(states=states, tokenizer=tokenizer)
    if kg_mode:
        if entity_index is None:
            entity_index = build_entity_index(tokenizer, kg.entities())
        cfg = dict(long_chains=False, eot=KG_CLOSE, avoid_duplicates=True,
                   forbid_revisits=True, sentinel=True)
        cfg.update(kg_kwargs or {})
        processor.add_pattern(KG_OPEN, KnowledgeGraphGeneration, index=entity_index,
                              get_relations=kg.properties, get_objects=kg.get, **cfg)
    if rule_mode:
        cfg = dict(close_text=RULE_CLOSE, vocab=SortedVocab.for_tokenizer(tokenizer))
        cfg.update(rule_kwargs or {})
        processor.add_pattern(RULE_OPEN, RuleGeneration, kg=kg, **cfg)
    return processor


# --------------------------------------------------------------------- trace
_BLOCK = re.compile(r"(<kg>.*?</kg>|<rule>.*?</rule>|<result>.*?</result>|<answer>.*?</answer>)",
                    re.DOTALL)
COLORS = {"free": "#00000000", "kg": "#cfe8ff", "rule": "#ffe3b3",
          "result": "#d9f2d0", "answer": "#eadcff"}
LABELS = {"kg": "KG mode", "rule": "RULE mode", "result": "Vadalog result",
          "answer": "answer"}


def segments(text):
    """Split generated text into (mode, text) pieces."""
    out = []
    for piece in _BLOCK.split(text):
        if not piece:
            continue
        for mode in ("kg", "rule", "result", "answer"):
            if piece.startswith("<%s>" % mode):
                out.append((mode, piece))
                break
        else:
            out.append(("free", piece))
    return out


@dataclass
class Trace:
    question: str
    text: str
    kg_paths: list = field(default_factory=list)
    rules: list = field(default_factory=list)       # RuleGeneration objects
    seconds: float = 0.0
    n_tokens: int = 0
    kg: object = None

    @property
    def segments(self):
        return segments(self.text)

    @property
    def answer(self):
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", self.text, re.DOTALL)
        return m.group(1) if m else None

    def relation_kind(self, relation):
        """'stored', 'derived' (Vadalog program) or 'rule' (written in this answer)."""
        kg = self.kg
        rel = relation[3:] if relation.startswith("is_") and relation[3:] in kg.schema() else relation
        if rel in kg.base:
            return "stored"
        if rel in dict(kg.session_rules):
            return "rule"
        return "derived"

    def summary(self):
        lines = ["question: %s" % self.question,
                 "%d tokens in %.1fs" % (self.n_tokens, self.seconds)]
        for path in self.kg_paths:
            for hop in path:
                lines.append("  <kg>   %s --%s [%s]--> %s" % (
                    hop.get("entity"), hop.get("relation"),
                    self.relation_kind(hop.get("relation", "")), hop.get("object")))
        for r in self.rules:
            res = r.result
            lines.append("  <rule> %s" % r.rule_text)
            if res is not None:
                lines.append("         -> %s (%.2fs Vadalog, %.2fs masking, %d tokens)"
                             % (res.summary(3), res.seconds, r.mask_seconds, len(r.steps)))
        lines.append("answer: %s" % self.answer)
        return "\n".join(lines)

    def _repr_html_(self):
        parts = []
        for mode, piece in self.segments:
            body = html.escape(piece)
            if mode == "free":
                parts.append(body)
            else:
                parts.append('<span title="%s" style="background:%s;border-radius:4px;'
                             'padding:1px 2px">%s</span>' % (LABELS[mode], COLORS[mode], body))
        legend = " ".join('<span style="background:%s;padding:1px 6px;border-radius:4px">%s</span>'
                          % (COLORS[m], LABELS[m]) for m in ("kg", "rule", "result", "answer"))
        rows = []
        for path in self.kg_paths:
            for hop in path:
                rel = str(hop.get("relation", ""))
                cells = (hop.get("entity"), rel, self.relation_kind(rel), hop.get("object"))
                rows.append("<tr><td>&lt;kg&gt;</td><td>%s</td><td><b>%s</b> <i>(%s)</i></td>"
                            "<td>%s</td></tr>" % tuple(html.escape(str(c)) for c in cells))
        for r in self.rules:
            res = r.result
            rows.append("<tr><td>&lt;rule&gt;</td><td colspan=2><code>%s</code></td><td>%s</td></tr>"
                        % (html.escape(r.rule_text or ""), html.escape(res.summary(3) if res else r.error or "")))
        table = ("<table style='font-size:90%%'><tr><th>mode</th><th>subject / rule</th>"
                 "<th>relation</th><th>object / Vadalog result</th></tr>%s</table>" % "".join(rows)
                 if rows else "")
        return ("<div style='font-family:monospace;white-space:pre-wrap;border:1px solid #ccc;"
                "padding:8px;border-radius:6px'><b>Q:</b> %s\n\n%s</div><div style='margin:4px 0'>%s"
                " &nbsp; %d tokens, %.1fs</div>%s"
                % (html.escape(self.question), "".join(parts), legend, self.n_tokens,
                   self.seconds, table))


class FullModeSession:
    """Model + tokenizer + KG + prompt: ``session.ask(question)`` -> Trace."""

    def __init__(self, model, tokenizer, kg, prompt_messages, entity_index=None,
                 kg_mode=True, rule_mode=True, rule_kwargs=None, kg_kwargs=None):
        self.model, self.tokenizer, self.kg = model, tokenizer, kg
        self.prompt_messages = prompt_messages
        self.processor = build_full_mode_processor(
            tokenizer, kg, entity_index=entity_index, kg_mode=kg_mode,
            rule_mode=rule_mode, rule_kwargs=rule_kwargs, kg_kwargs=kg_kwargs)

    def prompt(self, question, thinking=False):
        messages = list(self.prompt_messages) + [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    def ask(self, question, max_new_tokens=768, do_sample=False, thinking=False,
            keep_rules=False, streamer=None, constrained=True, **gen_kwargs):
        import torch
        if not keep_rules:
            self.kg.reset_rules()
        self.processor.reset_states()
        tok = raw_tokenizer(self.tokenizer)
        inputs = tok(self.prompt(question, thinking), return_tensors="pt").to(self.model.device)
        if do_sample:
            gen_kwargs = dict(dict(temperature=0.7, top_p=0.8, top_k=20), **gen_kwargs)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, logits_processor=[self.processor] if constrained else [],
                max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=1,
                streamer=streamer, **gen_kwargs)
        gen = out[0][inputs.input_ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        state = self.processor.states[0, 0]
        history = list(state.generation_history)
        if state.active_generation is not None:
            history.append(state.active_generation)
        paths, rules = [], []
        for g in history:
            if isinstance(g, KnowledgeGraphGeneration):
                paths.extend(g.generated_path_metadata)
            elif isinstance(g, RuleGeneration):
                rules.append(g)
        return Trace(question, text, paths, rules, time.time() - t0, len(gen), self.kg)
