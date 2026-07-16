# GCR Integration Branch

**Branch**: `gcr`

## Objective

Integrate ReFactX with Graph-Constrained Reasoning (GCR) to enable a single `model.generate()` call where:

1. **ReFactX** constrains entity name generation from an in-memory DictIndex (prefix tree of entity names from a knowledge graph)
2. **Seamlessly transitions** to **GCR** for graph path generation from the selected entity using a `MarisaTrie` built from Neo4j neighborhood paths

## Architecture

A single `ReFactXThenGCR` logits processor orchestrates two phases:

- **ENTITY phase**: Delegates to ReFactX's `ConstrainedLogitsProcessor` to generate a valid entity name from the index
- **Transition**: Detects entity completion and builds a GCR `MarisaTrie` from Neo4j
- **PATH phase**: GCR constrains remaining tokens to valid graph paths

## Key Files

- `refactx/generate.py` — patched `beam_permutation()` for transformers v4.49+ compatibility
- `refactx/index.py` — `DictIndex` used for entity name prefix matching
- Integration script at `../graph-constrained-reasoning/notebooks/refactx_neo4j_integration_v2.py`

## Status

- ReFactX dev branch cloned from `rpo19/ReFactX` (branch `develop`)
- Patched for transformers v4.49+ `beam_idx` 1D tensor handling
- Integration demo working with Qwen3.5-0.8B and Freebase in Neo4j
