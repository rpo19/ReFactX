# Task: verify the qwen36 count-tool numbers on the Mintaka mismatch cases

You are on a non-HPC machine with the `ReFactX` repository installed (or on
`PYTHONPATH`) and with access to the `qwen36` index. For each case below,
recompute what the `<count>` tool would return and compare it with:
1. the number observed in the RL eval trace (`trace_N`), and
2. the Mintaka ground truth (`GT`).

`tool == trace_N` means the tool/index is reproducible. `tool != GT` is the
semantic mismatch we are investigating (usually the wrong relation or a
relation whose KB fan-out is not the queried notion).

## Inputs

- `INDEX`: index URL. Postgres example:
  `postgresql://user:pass@host:5432/db?tablename=qwen36`
  (the table can also be passed with `TABLE=qwen36`).
  HTTP or a local trie file also work, see the script.
- `TOKENIZER`: the tokenizer used by the policy that produced the traces
  (default `Qwen/Qwen3.5-4B`). It must match the tokenizer used to build the
  index, otherwise token ids (and therefore counts) differ.
- `TABLE`: index table, default `qwen36`.

## Cases (Mintaka count questions, initial eval of run 4615736)

| # | Question | GT | Subject | Relation | trace_N |
|---|---|---|---|---|---|
| 1 | How many Academy Awards has Brad Pitt won? | 2 | Brad Pitt | award received | 15 |
| 2 | How many Pirates of the Caribbean films are there? | 5 | Pirates of the Caribbean | instance of | 1 |
| 3 | How many NBA championship titles did Michael Jordan win? | 6 | Michael Jordan | award received | 21 |
| 4 | How many video game consoles has Sega released? | 10 | Sega | has subsidiary | 10 |
| 5 | How many Call of Duty games have been made and released? | 16 | Call of Duty | has part(s) | 23 |
| 6 | How many Pulitzer Prize awards did Eugene O'Neill win? | 4 | Eugene O'Neill | award received | 4 |
| 7 | How many NBA championships did Kobe Bryant win? | 5 | Kobe Bryant | award received | 7 |
| 8 | How many NFL teams did John Elway play for? | 1 | John Elway | member of sports team | 2 |
| 9 | How many Mario Kart console games have there been? | 9 | Mario Kart | has part(s) | 12 |
| 10 | How many oceans are in the world? | 5 | ocean of Passion (2008 single by plantashun boiz) | instance of | 2 |

Case 10 is included because it shows a wrong subject chosen by the model, not a
wrong relation.

## Script

```python
import os
import refactx
from transformers import AutoTokenizer

INDEX_URL = os.environ["INDEX"]
TOKENIZER_NAME = os.environ.get("TOKENIZER", "Qwen/Qwen3.5-4B")
TABLE = os.environ.get("TABLE", "qwen36")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# load_index dispatches on the URL type. Postgres needs a table name; HTTP
# carries it in the URL; file/triple-list indexes need the tokenizer.
if INDEX_URL.startswith(("postgresql://", "postgres://")):
    index = refactx.load_index(INDEX_URL, tablename=TABLE)
elif INDEX_URL.startswith(("http://", "https://")):
    index = refactx.load_index(INDEX_URL)
else:
    index = refactx.load_index(INDEX_URL, tokenizer=tokenizer)


class _EmptyTree:
    def next_tokens(self, sequence):
        return {}, {}


class _State:
    def __init__(self):
        self.subtree_cache = _EmptyTree()

    def end_of_triple_reset(self):
        pass


def tool_count(subject, relation, show_children=False):
    # CountBranchesGeneration counts the prefix " <Subject> <Relation>" (the
    # leading space is part of the generated span after <count>).
    ids = tokenizer.encode(f" <{subject}> <{relation}>", add_special_tokens=False)
    state = _State()
    sp = getattr(index, "switch_parameter", None)
    if sp is not None and len(ids) >= sp - 1:
        # Warm the subtree cache with the truncated prefix, exactly like
        # CountBranchesGeneration._count_at_prefix.
        index.next_tokens(ids[: sp - 1], state=state)
    try:
        nxt, _ = index.next_tokens(ids, state=state)
    except TypeError:
        # Some index backends do not accept a state argument.
        nxt, _ = index.next_tokens(ids)
    if show_children:
        top = sorted(nxt.items(), key=lambda kv: -kv[1])[:8]
        print("   children:", [(tokenizer.decode([tid]), n) for tid, n in top])
    return sum(nxt.values())


CASES = [
    ("How many Academy Awards has Brad Pitt won?", "2", "Brad Pitt", "award received", 15),
    ("How many Pirates of the Caribbean films are there?", "5", "Pirates of the Caribbean", "instance of", 1),
    ("How many NBA championship titles did Michael Jordan win?", "6", "Michael Jordan", "award received", 21),
    ("How many video game consoles has Sega released?", "10", "Sega", "has subsidiary", 10),
    ("How many Call of Duty games have been made and released?", "16", "Call of Duty", "has part(s)", 23),
    ("How many Pulitzer Prize awards did Eugene O'Neill win?", "4", "Eugene O'Neill", "award received", 4),
    ("How many NBA championships did Kobe Bryant win?", "5", "Kobe Bryant", "award received", 7),
    ("How many NFL teams did John Elway play for?", "1", "John Elway", "member of sports team", 2),
    ("How many Mario Kart console games have there been?", "9", "Mario Kart", "has part(s)", 12),
    ("How many oceans are in the world?", "5", "ocean of Passion (2008 single by plantashun boiz)", "instance of", 2),
]

show = bool(os.environ.get("SHOW_CHILDREN"))
print(f"{'#':>2} {'GT':>4} {'trace':>5} {'tool':>5} {'tool==trace':>12} {'tool==GT':>8}  subject / relation")
for i, (question, gt, subject, relation, trace_n) in enumerate(CASES, 1):
    got = tool_count(subject, relation, show_children=show)
    print(
        f"{i:>2} {gt:>4} {trace_n:>5} {got:>5} "
        f"{str(got == trace_n):>12} {str(str(got) == str(gt)):>8}  "
        f"{subject} / {relation}"
    )
    if show:
        print("   question:", question)
```

Run it with `SHOW_CHILDREN=1` to also print the top object tokens the index
returns for each prefix.

## What to report

- A table with `GT`, `trace_N`, recomputed `tool`, and the two comparisons.
- Which cases have `tool != trace_N` (index/tokenizer/prefix mismatch) versus
  `tool == trace_N but tool != GT` (the tool is reproducible but counts the
  wrong notion).
- If `tool != trace_N`, check: same tokenizer as the run, leading space in the
  prefix, `switch_parameter` warming, and that the DB/table is the same
  `qwen36` snapshot.
