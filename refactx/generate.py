from transformers import LogitsProcessorList
from refactx.index import DictIndex
from transformers.generation.logits_process import LogitsProcessor
from transformers import ProcessorMixin
import torch
from copy import deepcopy
import math
import types
from dataclasses import dataclass, field
from typing import Type

from refactx.index import EmptyIndexException, TripleNotFoundException


def patch_model(model, verbose=True):
    _get_running_beams_for_next_iteration_original = model.__class__._get_running_beams_for_next_iteration
    
    def _get_running_beams_for_next_iteration_patch(self,*args, **kwargs):
        global CONSTRAINED_STATES
        running_sequences, running_beam_scores, running_beam_indices = _get_running_beams_for_next_iteration_original(self, *args, **kwargs)
        CONSTRAINED_STATES.beam_idx = running_beam_indices
        return running_sequences, running_beam_scores, running_beam_indices    
    
    model._get_running_beams_for_next_iteration = types.MethodType(_get_running_beams_for_next_iteration_patch, model)

    if verbose:
        print('WARNING: this patching method relies on shared mutable global state to support constrained generation with beam search. It is not thread-safe and may produce incorrect results in concurrent or multi-process setups (e.g. multiple workers).')


# ---------------------------------------------------------------------------
# PatternConfig — registers a pattern with its generation class
# ---------------------------------------------------------------------------

@dataclass
class PatternConfig:
    pattern: str
    generation_class: Type
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PatternConstrainedGeneration — base class for all constrained generation
# ---------------------------------------------------------------------------

class PatternConstrainedGeneration:
    """Base class for constrained generation logic.

    Subclasses implement ``constrain()`` which is called on every token
    while the generation is active.  Set ``self.done = True`` when the
    generation is complete; the state will move it to ``generation_history``.
    """

    def __init__(self, state, tokenizer, start_idx):
        self.state = state
        self.tokenizer = tokenizer
        self.start_idx = start_idx
        self.done = False

    def constrain(self, sequence, mask, mask_idx):
        raise NotImplementedError

    def on_complete(self):
        pass


# ---------------------------------------------------------------------------
# FactGeneration — trie-guided triple generation (was constrained_generation)
# ---------------------------------------------------------------------------

class FactGeneration(PatternConstrainedGeneration):
    """Generates ``<Subject> <Relation> <Object> .`` triples guided by a KB trie."""

    def __init__(self, state, tokenizer, start_idx,
                 index, sentinel=False, sentinel_text='no further records>',
                 avoid_duplicates=True, eot='\n'):
        super().__init__(state, tokenizer, start_idx)
        self.index = index
        self.sentinel = sentinel
        self.sentinel_text = sentinel_text
        self.avoid_duplicates = avoid_duplicates

        self.eot_tokens = []
        if eot is not None:
            if isinstance(eot, str):
                if isinstance(tokenizer, ProcessorMixin):
                    encoded = tokenizer.tokenizer.encode(eot, add_special_tokens=False)
                else:
                    encoded = tokenizer.encode(eot, add_special_tokens=False)
            else:
                encoded = list(eot)
            self.eot_tokens = encoded
        self.eot_remaining = []

        self._sentinel_ids_cache = None
        self.completed_with_sentinel = False
        self.prev_exhausted = set()

    def _sentinel_ids(self):
        if self._sentinel_ids_cache is None:
            if isinstance(self.tokenizer, ProcessorMixin):
                self._sentinel_ids_cache = self.tokenizer.tokenizer.encode(
                    self.sentinel_text, add_special_tokens=False)
            else:
                self._sentinel_ids_cache = self.tokenizer.encode(
                    self.sentinel_text, add_special_tokens=False)
        return self._sentinel_ids_cache

    def _begin_sentinel(self, mask, mask_idx, sequence=None, sentinel_text=None):
        """Start emitting sentinel tokens.

        Parameters
        ----------
        sentinel_text : str, optional
            Override the default ``self.sentinel_text`` for this invocation.
            Use this to emit a level-appropriate sentinel (e.g. with
            surrounding ``<>`` and ``.`` at subject/relation level).
        """
        if sentinel_text is not None:
            if isinstance(self.tokenizer, ProcessorMixin):
                ids = self.tokenizer.tokenizer.encode(
                    sentinel_text, add_special_tokens=False)
            else:
                ids = self.tokenizer.encode(
                    sentinel_text, add_special_tokens=False)
        else:
            ids = self._sentinel_ids()
        mask[mask_idx, :] = -math.inf
        mask[mask_idx, ids[0]] = 0
        self.state.sentinel_remaining = list(ids[1:])
        self.completed_with_sentinel = True
        if sequence is not None:
            self._sentinel_triple = sequence[:]
        if not self.state.sentinel_remaining:
            self._finish()

    def _finish(self):
        if self.completed_with_sentinel and hasattr(self, '_sentinel_triple'):
            self.state.cache_add(self._sentinel_triple, self.start_idx)
        self.state.subtree_cache.reset()
        self.state.sentinel_remaining = []
        self.state.state = 0
        self.done = True

    def constrain(self, sequence, mask, mask_idx):
        state = self.state

        # Mid-sentinel: keep emitting the "no further records" object token by token.
        if state.sentinel_remaining:
            nxt = state.sentinel_remaining.pop(0)
            mask[mask_idx, :] = -math.inf
            mask[mask_idx, nxt] = 0
            if not state.sentinel_remaining:
                self._finish()
            return

        if self.eot_remaining:
            nxt = self.eot_remaining.pop(0)
            mask[mask_idx, :] = -math.inf
            mask[mask_idx, nxt] = 0
            if not self.eot_remaining:
                self._finish()
            return

        possible_tokens, _ = self.index.next_tokens(sequence, state=state)

        if not self.sentinel:
            # ---- original behaviour: prune exhausted branches ----
            if self.avoid_duplicates:
                try:
                    visited_tokens, _ = state.cache_index.next_tokens(sequence)
                    state.cache_index.subtract_tokens(possible_tokens, visited_tokens)
                except EmptyIndexException:
                    pass
                except TripleNotFoundException:
                    pass

            possible_tokens = list(possible_tokens.keys())

            if len(possible_tokens) == 0:
                state.cache_add(sequence, self.start_idx)
                if self.eot_tokens:
                    mask[mask_idx, :] = -math.inf
                    mask[mask_idx, self.eot_tokens[0]] = 0
                    self.eot_remaining = list(self.eot_tokens[1:])
                    if not self.eot_remaining:
                        self._finish()
                    return
                self._finish()
            else:
                vocab_size = mask.shape[-1]
                invalid_tokens = [t for t in possible_tokens if t < 0 or t >= vocab_size]
                if invalid_tokens:
                    raise ValueError(
                        f"Invalid token ids in constrained generation: {invalid_tokens[:10]} "
                        f"(showing up to 10) with vocab_size={vocab_size}"
                    )

            mask[mask_idx, possible_tokens] = 0
            return

        # ---- sentinel-enabled behaviour ----
        # If the model just picked an exhausted token (remaining == 0)
        # on the previous step, fire the sentinel — there are no valid
        # children left down this branch.
        if sequence and sequence[-1] in self.prev_exhausted:
            in_object = (self.tokenizer or state.tokenizer).decode(
                sequence).count('> <') >= 2
            if in_object:
                self._begin_sentinel(mask, mask_idx, sequence=sequence)
            else:
                self._begin_sentinel(
                    mask, mask_idx, sequence=sequence,
                    sentinel_text=' <no further records> .')
            return

        # Compute *live* (remaining > 0) and *exhausted* (remaining == 0)
        # dicts.  Over-visited tokens (remaining < 0) are discarded.
        # Both live and exhausted are always offered — the prev_exhausted
        # check above catches when the model enters a dead end.
        live = {}
        exhausted = {}
        if self.avoid_duplicates:
            visited_tokens = {}
            try:
                visited_tokens, _ = state.cache_index.next_tokens(sequence)
            except EmptyIndexException:
                pass
            except TripleNotFoundException:
                pass
            for tok, total in possible_tokens.items():
                remaining = total - visited_tokens.get(tok, 0)
                if remaining > 0:
                    live[tok] = remaining
                elif remaining == 0:
                    exhausted[tok] = remaining
        else:
            live = dict(possible_tokens)

        self.prev_exhausted = set(exhausted.keys())

        if len(possible_tokens) == 0:
            # Index itself has nothing at this prefix — record and finish.
            state.cache_add(sequence, self.start_idx)
            self._finish()
            mask[mask_idx, :] = 0
            return

        # At subject / relation levels offer live + exhausted so the
        # model can explore exhausted branches (prev_exhausted catches
        # it on the next call).  At object+ levels only live tokens
        # are offered — exhausted objects must not be re-chosen.
        _decoded = (self.tokenizer or state.tokenizer).decode(sequence)
        at_or_past_object = _decoded.count('>') >= 2

        if at_or_past_object:
            offer_keys = list(live.keys())
        else:
            offer_keys = list(set(list(live.keys()) + list(exhausted.keys())))

        if offer_keys:
            vocab_size = mask.shape[-1]
            valid_keys = [t for t in offer_keys if 0 <= t < vocab_size]
            mask[mask_idx, valid_keys] = 0
        else:
            # ALL children over-visited (remaining < 0) — fire sentinel.
            in_object = (self.tokenizer or state.tokenizer).decode(
                sequence).count('> <') >= 2
            if in_object:
                self._begin_sentinel(mask, mask_idx, sequence=sequence)
            else:
                self._begin_sentinel(
                    mask, mask_idx, sequence=sequence,
                    sentinel_text=' <no further records> .')


# ---------------------------------------------------------------------------
# CountBranchesGeneration — count_branches: <S> <R> tool call
# ---------------------------------------------------------------------------

class CountBranchesGeneration(PatternConstrainedGeneration):
    """Handles ``count_branches: <Subject> <Relation>`` tool calls.

    Triggered when the state detects ``count_branches: ``.  Collects prefix
    tokens until a newline, counts KB leaves, then forces ``= <count>``.
    """

    MODE_COLLECTING = 0
    MODE_EMITTING = 1

    def __init__(self, state, tokenizer, start_idx, kb_index, delimiter="\n"):
        super().__init__(state, tokenizer, start_idx)
        self.kb_index = kb_index
        self.mode = self.MODE_COLLECTING
        self.prefix_tokens: list[int] = []
        self.count_tokens: list[int] = []
        self.calls: list[tuple[str, int]] = []
        encoded_delim = self._encode(delimiter)
        self.delimiter_token = encoded_delim[0] if encoded_delim else None

    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "tokenizer"):
            return self.tokenizer.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids)

    def _count_at_prefix(self, prefix_token_ids: list[int]) -> int:
        import logging
        _log = logging.getLogger(__name__)
        try:
            # For PostgresTrieIndex: if the prefix exceeds switch_parameter,
            # the index can't resolve it in one shot — it needs a warmed-up
            # subtree cache.  Warm it by querying with the truncated prefix
            # first, then query with the full prefix.
            sp = getattr(self.kb_index, 'switch_parameter', None)
            if sp is not None and len(prefix_token_ids) >= sp - 1:
                warm = prefix_token_ids[:sp - 1]
                self.kb_index.next_tokens(warm, state=self.state)

            next_tokens, _ = self.kb_index.next_tokens(
                prefix_token_ids, state=self.state)
        except Exception as exc:
            _log.warning(
                "count_branches: lookup failed for prefix %s: %s",
                prefix_token_ids, exc)
            return 0
        if not next_tokens:
            _log.warning(
                "count_branches: no children for prefix %s (%r)",
                prefix_token_ids,
                self.tokenizer.tokenizer.decode(prefix_token_ids)
                    if hasattr(self.tokenizer, 'tokenizer')
                    else prefix_token_ids)
            return 0
        return sum(next_tokens.values())

    def constrain(self, sequence, mask, mask_idx):
        last_token = sequence[-1] if sequence else None

        if self.mode == self.MODE_COLLECTING:
            if self.delimiter_token is not None and last_token == self.delimiter_token:
                count = self._count_at_prefix(self.prefix_tokens)
                prefix_text = self._decode(self.prefix_tokens).strip()
                self.calls.append((prefix_text, count))
                count_str = f" = {count}"
                self.count_tokens = self._encode(count_str)
                self.mode = self.MODE_EMITTING
                self._emit_count(mask, mask_idx)
            else:
                if last_token is not None:
                    self.prefix_tokens.append(last_token)
                mask[mask_idx, :] = 0

        elif self.mode == self.MODE_EMITTING:
            self._emit_count(mask, mask_idx)

    def _emit_count(self, mask, mask_idx):
        if self.count_tokens:
            nxt = self.count_tokens.pop(0)
            mask[mask_idx, :] = -math.inf
            if 0 <= nxt < mask.shape[-1]:
                mask[mask_idx, nxt] = 0
            if not self.count_tokens:
                self.done = True
        else:
            self.done = True


# ---------------------------------------------------------------------------
# PatternConstrainedState — per-beam state machine
# ---------------------------------------------------------------------------

class PatternConstrainedState():
    def __init__(self, tokenizer, cache_index, subtree_cache, state=0,
                 debug=False, regex_window=10, ignore_case=True,
                 thinking_end_pattern='</think>') -> None:

        self.NORMAL_GENERATION = 0
        self.CONSTRAINED_GENERATION = 1

        self.token_ids = []
        self.input_ids = []

        self.tokenizer = tokenizer
        self.regex_window = regex_window

        self.ignore_case = ignore_case

        self.patterns: list[PatternConfig] = []
        self.active_generation: PatternConstrainedGeneration | None = None
        self.generation_history: list[PatternConstrainedGeneration] = []

        self.state = state
        self.cursor = 0

        self.cache_index = cache_index
        self.generated_triples = []
        self.generated_triples_idx = []
        self.generated_triples_str = []

        self.subtree_cache = subtree_cache

        self.sentinel_remaining = []

        self.thinking_end_pattern = thinking_end_pattern.lower() if ignore_case and thinking_end_pattern else thinking_end_pattern
        self._thinking_cache_reset_done = False

        self._first_call = True

        self.debug = debug
        self.debug_history = []

    def print_debug(self, tokenizer, print_class=False, end_with_newline=True):
        if print_class:
            print('{} '.format(self), end='')
        if end_with_newline:
            print()

    def first_call(self):
        if self._first_call:
            self._first_call = False
            return True
        else:
            return False

    def cache_add(self, sequence, start_idx=0):
        self.cache_index.add(sequence, new_leaf=True)
        self.generated_triples.append(sequence)
        self.generated_triples_idx.append([list(range(start_idx, len(sequence)))])
        if self.tokenizer is not None:
            self.generated_triples_str.append(self.tokenizer.decode(sequence))

    def is_constrained(self):
        return self.active_generation is not None

    def end_of_triple_reset(self):
        self.subtree_cache.reset()
        self.sentinel_remaining = []
        self.state = 0

    def reset(self):
        self.state = 0
        self.token_ids = []
        self.input_ids = []
        self.history = ()
        self.cursor = 0
        self.generated_triples = []
        self.cache_index.reset()
        self._thinking_cache_reset_done = False
        self.active_generation = None
        self.patterns = []
        self.generation_history = []
        self.end_of_triple_reset()

    def copy(self, other, copy=True):
        self.state = other.state
        self.tokenizer = other.tokenizer
        self.regex_window = other.regex_window
        self.token_ids = deepcopy(other.token_ids)
        self.input_ids = deepcopy(other.input_ids)

        self.history = other.history
        self.cursor = other.cursor

        self.patterns = other.patterns
        self.active_generation = deepcopy(other.active_generation) if copy and other.active_generation else other.active_generation
        self.generation_history = deepcopy(other.generation_history) if copy else other.generation_history

        self.generated_triples = other.generated_triples.copy() if copy else other.generated_triples
        self.cache_index = deepcopy(other.cache_index) if copy else other.cache_index
        self.subtree_cache = deepcopy(other.subtree_cache) if copy else other.subtree_cache
        self.thinking_end_pattern = other.thinking_end_pattern
        self._thinking_cache_reset_done = other._thinking_cache_reset_done

        self.debug = other.debug
        self.debug_history = deepcopy(other.debug_history) if copy else other.debug_history

    def _decode_recent(self):
        text = self.tokenizer.decode(self.token_ids[-self.regex_window:])
        if self.ignore_case:
            text = text.lower()
        return text

    def _activate_generation(self, config: PatternConfig):
        start_idx = self.get_cursor()
        self.active_generation = config.generation_class(
            state=self, tokenizer=self.tokenizer, start_idx=start_idx,
            **config.config
        )
        self.state = self.CONSTRAINED_GENERATION
        self.token_ids = []
        self.cursor = 0

    def update(self, new_token):
        state = self.state
        self.token_ids.append(new_token)
        self.input_ids.append(new_token)
        self.cursor += 1

        # If a generation is active, check for completion
        if self.active_generation:
            if self.active_generation.done:
                self.active_generation.on_complete()
                self.generation_history.append(self.active_generation)
                self.active_generation = None
                self.end_of_triple_reset()
                self.token_ids = []
            else:
                return

        # No active generation: check patterns
        text = self._decode_recent()
        for config in self.patterns:
            pattern = config.pattern.lower() if self.ignore_case else config.pattern
            if text.endswith(pattern):
                self._activate_generation(config)
                break

        # Feat 3: detect end-of-thinking and reset duplicate cache
        if (not self._thinking_cache_reset_done and
                self.thinking_end_pattern is not None):
            normalized = "".join(text.split())
            if normalized.rstrip().endswith(self.thinking_end_pattern):
                self.cache_index.reset()
                self._thinking_cache_reset_done = True

        if self.debug:
            self.debug_history.append({
                'state': self.state,
                'token': new_token,
                'token_ids': deepcopy(self.token_ids),
                'input_ids': deepcopy(self.input_ids),
            })

    def _update_state(self, state, initial_cursor=0):
        if state != self.state:
            self.history = (self.state, self.cursor)
            self.state = state
            self.cursor = initial_cursor
            self.token_ids = []

    def get_cursor(self):
        return self.cursor


# ---------------------------------------------------------------------------
# ConstrainedStateList — 2D container [batch][beam]
# ---------------------------------------------------------------------------

class ConstrainedStateList():
    def __init__(self, states, num_beams=1, num_batches=1, debug=False, debug_tokenizer=None):
        if states == 'auto':
            assert debug_tokenizer is not None, 'debug_tokenizer must be provided when states is "auto"'
            states = []
            for _ in range(num_batches):
                batch_states = []
                for _ in range(num_beams):
                    batch_states.append(PatternConstrainedState(
                        tokenizer=debug_tokenizer,
                        cache_index=DictIndex(),
                        subtree_cache=DictIndex(),
                    ))
                states.append(batch_states)
        elif states != []:
            assert isinstance(states, list) and isinstance(states[0], list), 'ERROR: states is not a list of lists'
            assert len(states) == num_batches and len(states[0]) == num_beams, 'ERROR: states size does not match num_batches or num_beams'
        
        self.states = states
            
        self.num_beams = num_beams
        self.num_batches = num_batches
        self.beam_idx = []

        self.debug = debug
        self.debug_tokenizer = debug_tokenizer
        assert not self.debug or self.debug_tokenizer is not None

        self.num_permutations = 0

    def __getitem__(self, key):
        if isinstance(key, tuple):
            batch_key, beam_key = key

            if isinstance(batch_key, int) and isinstance(beam_key, int):
                return self.states[batch_key][beam_key]
            else:
                if isinstance(batch_key, int):
                    batch_key = slice(batch_key, batch_key + 1)
                if isinstance(beam_key, int):
                    beam_key = slice(beam_key, beam_key + 1)

                sliced_states = [row[beam_key] for row in self.states[batch_key]]

                new_num_batches = len(sliced_states)
                new_num_beams = len(sliced_states[0]) if sliced_states else 0

                return ConstrainedStateList(
                    sliced_states,
                    num_beams=new_num_beams,
                    num_batches=new_num_batches
                )

        elif isinstance(key, slice):
            return ConstrainedStateList(
                self.states[key],
                num_beams=self.num_beams,
                num_batches=len(self.states[key])
            )
        elif isinstance(key, int):
            return self.states[key]
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    def __len__(self):
        return len(self.states) * len(self.states[0])

    def reset(self):
        for batch in self.states:
            for state in batch:
                state.reset()

    def get_batch_idx(self, idx):
        return int(idx // self.num_beams)
    
    def get_beam_idx(self, idx):
        return int(idx % self.num_beams)
    
    def get_last_beam_z(self):
        z = -1
        ids = (self.beam_idx[0, 0] != -1).nonzero(as_tuple=False).squeeze()
        if ids.numel() > 0:
            if ids.dim() == 0:
                z = ids.item()
            else:
                z = ids[-1].item()
        return z

    def beam_permutation(self):
        if len(self.beam_idx) > 0:
            assert self.beam_idx.shape[0] * self.beam_idx.shape[1] == self.num_beams * self.num_batches, f'ERROR: beam_idx size unexpected: {len(self.beam_idx)} != {self.num_beams} * {self.num_batches}'
            copies = []
            for batch_i in range(self.num_batches):
                batch_copies = []
                for beam_i in range(self.num_beams):
                    batch_copies.append(self[batch_i, beam_i].dump())
                copies.append(batch_copies)
            last_beam_z = self.get_last_beam_z()
            if last_beam_z >= 0:
                if self.debug:
                    for batch_idx in range(self.beam_idx.shape[0]):
                        for num_beam in range(self.beam_idx.shape[1]):
                            print((batch_idx, num_beam), end='')
                            self.states[batch_idx][num_beam].print_debug(self.debug_tokenizer)

                for batch_idx in range(self.beam_idx.shape[0]):
                    for num_beam in range(self.beam_idx.shape[1]):
                        replacement_idx = self.beam_idx[batch_idx, num_beam, last_beam_z]
                        replacement_batch_idx = self.get_batch_idx(replacement_idx)
                        local_beam_idx = self.get_beam_idx(replacement_idx)
                        assert replacement_batch_idx == batch_idx, f'ERROR: permutating between different batches! {replacement_batch_idx} --> {batch_idx}, with num_beams {self.num_beams}. replacement_idx {replacement_idx}'
                        if num_beam != local_beam_idx:
                            self.states[batch_idx][num_beam].load(copies[batch_idx][local_beam_idx], copy=True)
                            self.num_permutations += 1
                            if self.debug:
                                print(f'permutation {self.num_permutations}: ({batch_idx},{local_beam_idx}) into {batch_idx}{num_beam}')


# ---------------------------------------------------------------------------
# ConstrainedLogitsProcessor — thin orchestrator
# ---------------------------------------------------------------------------

class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, states, tokenizer, reinit_states=False):
        self.states = states
        self.reinit_states = reinit_states
        self.tokenizer = tokenizer

    def add_pattern(self, pattern, generation_class, **config):
        cfg = PatternConfig(pattern=pattern, generation_class=generation_class, config=config)
        for batch in self.states.states:
            for state in batch:
                state.patterns.append(cfg)

    def _reinit_states(self, num_beams, num_batches):
        self.states.__init__('auto',
                num_beams=num_beams,
                num_batches=num_batches,
                debug_tokenizer=self.tokenizer
        )

    def _reinit_states_to_input_ids(self, input_ids):
        for i in range(input_ids.shape[0]):
            batch_idx = self.states.get_batch_idx(i)
            beam_i = self.states.get_beam_idx(i)
            self.states[batch_idx, beam_i].input_ids = input_ids[i].tolist()[:-1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if input_ids.shape[0] != len(self.states):
            message = f'number of states ({len(self.states)}) should match `num_batches * num_beams` ({input_ids.shape[0]})'
            if self.reinit_states:
                num_beams = 1
                num_batches = input_ids.shape[0]
                self._reinit_states(num_beams, num_batches)
                print(f'Warning: {message}')
            else:
                raise ValueError(f'Error: {message}')
            self._reinit_states(num_beams=1, num_batches=input_ids.shape[0])
            
        self.states.beam_permutation()

        mask = torch.zeros_like(scores)
        mask = mask.to(scores.dtype)

        for i in range(input_ids.shape[0]):
            batch_idx = self.states.get_batch_idx(i)
            beam_i = self.states.get_beam_idx(i)
            sequence = input_ids[i].tolist()

            if not self.states[batch_idx, beam_i].first_call():
                last_token = sequence[-1]
                if sequence[:-1] != self.states[batch_idx, beam_i].input_ids:
                    if self.reinit_states:
                        print(f'Warning: sequence changed unexpectedly for batch {batch_idx} beam {beam_i}, reinitializing all states.')
                        self.states.reset()
                        self._reinit_states_to_input_ids(input_ids)
                        self.states[batch_idx, beam_i].update(last_token)
                    else:
                        raise ValueError(f'Error: sequence changed unexpectedly for batch {batch_idx} beam {beam_i}')
                else:
                    self.states[batch_idx, beam_i].update(last_token)
            else:
                self.states[batch_idx, beam_i].input_ids = sequence

            if self.states[batch_idx, beam_i].is_constrained():
                mask[i] = -math.inf
                state = self.states[batch_idx, beam_i]
                constrain_generation_sequence_start_idx = len(sequence) - state.get_cursor()
                constrain_generation_sequence = sequence[constrain_generation_sequence_start_idx:]
                state.active_generation.constrain(
                    constrain_generation_sequence, mask, i)

        scores_processed = scores + mask

        return scores_processed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_constrained_logits_processor(tokenizer, index, num_beams=1, num_batches=1,
                                     return_list=True, sentinel=False,
                                     fact_pattern='<fact>', count_pattern='<count>',
                                     eot='</fact>\n', **kwargs):
    CONSTRAINED_STATES.__init__('auto',
                num_beams=num_beams,
                num_batches=num_batches,
                debug_tokenizer=tokenizer
        )

    constrained_processor = ConstrainedLogitsProcessor(
        states=CONSTRAINED_STATES, tokenizer=tokenizer)

    fact_config = dict(kwargs)
    fact_config['eot'] = eot
    constrained_processor.add_pattern(
        fact_pattern, FactGeneration,
        index=index, sentinel=sentinel, **fact_config)

    constrained_processor.add_pattern(
        count_pattern, CountBranchesGeneration,
        kb_index=index)

    if return_list:
        logits_processor_list = LogitsProcessorList([
            constrained_processor
        ])
        return logits_processor_list
    else:
        return constrained_processor


CONSTRAINED_STATES = ConstrainedStateList([])
