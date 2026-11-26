# DDB Graph Data

This directory contains the preprocessed biomedical knowledge graph used by
the MedQA-USMLE data.

The data and preprocessing originate from the QAGNN repository associated with
the paper:

- Repository: <https://github.com/michiyasunaga/qagnn/tree/main>
- MedQA-USMLE preprocessing notebook:
  <https://github.com/michiyasunaga/qagnn/blob/main/utils_biomed/preprocess_medqa_usmle.ipynb>

## Reproduction Workflow

1. Download the preprocessed biomedical data archive referenced by QAGNN:
   <https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_biomed.zip>
2. Unzip the archive. It should produce the `data_preprocessed_biomed`
   directory containing this `ddb` directory.
3. Run `utils/verbalize_medqa_graph.py` from the refactx repository, pointing
   it at the extracted graph files:

```bash
python utils/verbalize_medqa_graph.py \
  --graph /path/to/data_preprocessed_biomed/ddb/ddb.graph \
  --ptrs /path/to/data_preprocessed_biomed/ddb/ptrs.txt \
  --vocab /path/to/data_preprocessed_biomed/ddb/vocab.txt \
  --output /path/to/medqa_usmle_triples.txt
```

The script produces angle-bracket triples with human-readable entity and
relation labels.

## Files

- `ddb.graph`: Pickled NetworkX `MultiDiGraph`. Nodes are integer indices into
  `vocab.txt`; edges are directed and store a numeric `rel` relation ID and a
  `weight` attribute. Reverse edges are also stored.
- `ddb_relas.json`: JSON dictionary of raw relation records. Each value is a
  three-element list of numeric/string IDs from the preprocessing pipeline.
- `vocab.txt`: Entity labels, one label per line. The line number is the graph
  node index (zero-based).
- `ptrs.txt`: Original entity IDs, one per line, aligned with `vocab.txt`.
  IDs may be numeric or UMLS CUIs such as `C0002965`.
- `ddb_names.json`: Alternate entity-name mapping. Each key is a name and its
  value contains an original entity ID and a type/category flag.
- `ddb_to_umls_cui.txt`: Mapping from DDB entities to UMLS CUI identifiers.
- `ent_emb.npy`: NumPy entity embedding matrix. Rows align with
  `vocab.txt`; the supplied matrix has 768-dimensional embeddings.

## Graph Counts

The serialized `ddb.graph` contains 9,956 nodes and 99,948 directed edges.
`ddb_relas.json` contains 49,974 raw relation records. The larger edge count
comes from storing relations in both directions. The metadata files contain
9,958 entity rows, so two metadata entities are not present as graph nodes.

## Relation Labels

The graph stores relation IDs numerically (`rel=0` through `rel=29`). The
original graph-construction notebook defines IDs `0` through `14` as the
following forward relations:

The relation dictionary is taken from the QAGNN MedQA-USMLE preprocessing
notebook linked above.

```text
0  belongs_to_the_category_of
1  is_a_category
2  may_cause
3  is_a_subtype_of
4  is_a_risk_factor_of
5  is_associated_with
6  may_contraindicate
7  interacts_with
8  belongs_to_the_drug_family_of
9  belongs_to_drug_super-family
10 is_a_vector_for
11 may_be_allelic_with
12 see_also
13 is_an_ingredient_of
14 may_treat
```

IDs `15` through `29` are reverse edges corresponding to IDs `0` through
`14`. An edge can therefore be rendered as:

```text
<Dominant R wave in V1> <is_a_subtype_of> <EKG abnormalities> .
```

The source notebook spells relation 13 as `is_an_ingradient_of`; this project
corrects that typo when verbalizing triples to `is_an_ingredient_of`.
