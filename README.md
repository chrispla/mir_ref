<img src="docs/img/mir_ref_logo.svg" align="left" height="110">

# mir_ref

Representation Evaluation Framework for Music Information Retrieval tasks | [Paper](https://arxiv.org/abs/2312.05994)

`mir_ref` is an open-source library for evaluating audio representations (embeddings or others) on a variety of music-related downstream tasks and datasets. It provides ready-to-use tasks, datasets, deformations, embedding models, and downstream models for config-based, no-code experiment orchestration. Components are modular, so it's easy to add custom embedding models, datasets, metrics, etc. Audio-specific results analysis and visualization tools are also provided.

## Installation

Clone the repository. Create and activate a python>3.9 environment and install `mir_ref`.

```
cd mir_ref
pip install -e .
```

## Example results
We conducted an example evaluation of 7 models in 6 tasks with 4 deformations and 5 different probing setups. For the full results, please refer to the 'Evaluation' chapter of [this thesis document](https://zenodo.org/records/8380471), pages 39-58.

## Citing
If you use `mir_ref`, please cite the following [paper](https://arxiv.org/abs/2312.05994):
```
@inproceedings{mir_ref,
    author = {Christos Plachouras and Pablo Alonso-Jim\'enez and Dmitry Bogdanov},
    title = {mir_ref: A Representation Evaluation Framework for Music Information Retrieval Tasks},
    booktitle = {37th Conference on Neural Information Processing Systems (NeurIPS), Machine Learning for Audio Workshop},
    address = {New Orleans, LA, USA},
    year = 2023,
}
```
