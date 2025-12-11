ReFactX is a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. It uses constrained generation with a pre-built prefix-tree index. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact.

<!--- BADGES: START --->
[![ISWC 2025](https://img.shields.io/badge/ISWC-2025-brightgreen)][#iswc-paper-package]
[![Arxiv](https://img.shields.io/badge/arXiv-2508.16983-B31B1B.svg)][#arxiv-paper-package]

[#iswc-paper-package]: https://doi.org/10.1007/978-3-032-09527-5_16
[#arxiv-paper-package]: https://arxiv.org/abs/2508.16983

<!--- BADGES: END --->

## Install and Try

```bash
pip install refactx
```

Quick example (print package version)

```python
import refactx
print(refactx.__version__)
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
