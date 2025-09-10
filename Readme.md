# ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation
### Riccardo Pozzi, Matteo Palmonari, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati

<!--- BADGES: START --->
[![Arxiv](https://img.shields.io/badge/arXiv-2508.16983-B31B1B.svg)][#arxiv-paper-package]
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)][#license-gh-package]

[#arxiv-paper-package]: https://arxiv.org/abs/2508.16983
[#license-gh-package]: LICENSE

<!--- BADGES: END --->

The paper has been accepted at ISWC 2025. A preprint that has not
undergone peer review is available at
[https://arxiv.org/abs/2508.16983](https://arxiv.org/abs/2508.16983).

We present ReFactX, a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from Wikidata are verbalized in 800 million textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact.


This repository contains the source code for using ReFactX and reproducing our work accepted at ISWC 2025.

## Setup
- install the requirements `pip install -r requirements.txt`
- prepare the `.env` file: `cp env-sample.txt .env`, then edit `.env` (can be skipped if using the simple index in the try_refactx notebook)

## Try ReFactX
For quickly trying ReFactX with an in-memory prefix tree (derived from a 31k-facts knowledge base) use the notebook [try_refactx.ipynb](notebooks/try_refactx.ipynb).

## Wikidata Prefix Tree
Refer to [PrefixTree.md](PrefixTree.md) for creating the Wikidata prefix tree we used in our work.

## Experiments
To reproduce our experiments use the `eval.py` script replacing INDEX, MODEL, and DATASET according to your needs (each of them is a python file to import).
```
python eval.py --index INDEX --model MODEL --dataset DATASET
```

### Throughtput
For the throughput experiment run
```
python throughput.py --model MODEL --index INDEX --max-tokens 4001 --output out.json [--unconstrained-generation]
```

### Cite
```
@misc{pozzi2025refactxscalablereasoningreliable,
      title={ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation}, 
      author={Riccardo Pozzi and Matteo Palmonari and Andrea Coletta and Luigi Bellomarini and Jens Lehmann and Sahar Vahdati},
      year={2025},
      eprint={2508.16983},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.16983},
      doi = "10.48550/arXiv.2508.16983",
}
```