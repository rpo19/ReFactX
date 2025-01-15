## Filter labels from the dump
We filter for labels, altLabels, and descriptions

```
bzgrep -P '(http://www\\.w3\\.org/2000/01/rdf-schema#label|http://www\\.w3\\.org/2004/02/skos/core#altLabel|http://schema\\.org/description).*\\@en\s+.' latest-truthy.nt.bz2 | pv | gzip -c > /workspace/data/latest-truthy-labels-descriptions.nt.gz
```