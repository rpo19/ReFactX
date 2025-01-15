## Filter labels from the dump
We filter for labels and altLabels.

```
bzgrep -P '(http://www\\.w3\\.org/2000/01/rdf-schema#label|http://www\\.w3\\.org/2004/02/skos/core#altLabel).*\\@en\s+.' ../data/latest-truthy.nt.bz2 | pv | gzip -c > /workspace/data/latest-truthy-labels.nt.gz
```