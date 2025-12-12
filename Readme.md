# ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation
### Riccardo Pozzi, Matteo Palmonari, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati

<!--- BADGES: START --->
[![ISWC 2025](https://img.shields.io/badge/ISWC-2025-brightgreen)][#iswc-paper-package]
[![Arxiv](https://img.shields.io/badge/arXiv-2508.16983-B31B1B.svg)][#arxiv-paper-package]
[![HF](https://img.shields.io/badge/Hugging%20Face-Dataset-orange?logo=huggingface)][#hf-dataset]
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)][#license-gh-package]

[#iswc-paper-package]: https://doi.org/10.1007/978-3-032-09527-5_16
[#arxiv-paper-package]: https://arxiv.org/abs/2508.16983
[#hf-dataset]: https://huggingface.co/datasets/rpozzi/ReFactX_data
[#license-gh-package]: LICENSE

<!--- BADGES: END --->

The implementation corresponding to the ISWC 2025 paper is available at the [ISWC2025 branch](https://github.com/rpo19/ReFactX/tree/ISWC2025).


A preprint that has not
undergone peer review is available at
[https://arxiv.org/abs/2508.16983](https://arxiv.org/abs/2508.16983).


We present ReFactX, a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from Wikidata are verbalized in 800 million textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact.

![ReFactX Example](misc/refactx_example.png)

This repository contains the source code for using ReFactX and reproducing our work accepted at ISWC 2025.

## Install ReFactX
Create a virtualenv or conda environment, then
```
pip install refactx
```

## Try ReFactX
For quickly trying ReFactX with an in-memory prefix tree (derived from a 31k-facts knowledge base) use the notebook [try_refactx.ipynb](notebooks/try_refactx.ipynb).

## Development Setup
- install the requirements `pip install -r requirements.txt`
- prepare the `.env` file: `cp env-sample.txt .env`, then edit `.env` (can be skipped if using the simple index in the try_refactx notebook)


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
@InProceedings{10.1007/978-3-032-09527-5_16,
      author="Pozzi, Riccardo
      and Palmonari, Matteo
      and Coletta, Andrea
      and Bellomarini, Luigi
      and Lehmann, Jens
      and Vahdati, Sahar",
      editor="Garijo, Daniel
      and Kirrane, Sabrina
      and Salatino, Angelo
      and Shimizu, Cogan
      and Acosta, Maribel
      and Nuzzolese, Andrea Giovanni
      and Ferrada, Sebasti{\'a}n
      and Soulard, Thibaut
      and Kozaki, Kouji
      and Takeda, Hideaki
      and Gentile, Anna Lisa",
      title="ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation",
      booktitle="The Semantic Web -- ISWC 2025",
      year="2026",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="290--308",
      isbn="978-3-032-09527-5",
      doi="10.1007/978-3-032-09527-5_16",
      url="https://doi.org/10.1007/978-3-032-09527-5_16"
}
```