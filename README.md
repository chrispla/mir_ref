<img src="docs/img/mir_ref_logo.svg" align="left" height="110">

# mir_ref

Representation Evaluation Framework for Music Information Retrieval tasks | [Paper](https://arxiv.org/abs/2312.05994)

`mir_ref` is an open-source library for evaluating audio representations (embeddings or others) on a variety of music-related downstream tasks and datasets. It provides ready-to-use tasks, datasets, deformations, embedding models, and downstream models for config-based, no-code experiment orchestration. Components are modular, so it's easy to add custom embedding models, datasets, metrics, etc. Audio-specific results analysis and visualization tools are also provided.

## Disclaimer
Many fixes and documentation improvements are expected soon.

## Setup

Clone the repository. Create and activate a python>3.9 environment and install the requirements.

```
cd mir_ref
pip install -r requirements.txt
```

## Running
`mir_ref` is comprised of 4 main functions-commands: `deform`, for generating deformations from a dataset; `extract`, for extracting features; `train`, for training probes; and `evaluate`, for evaluating them. These can be run as follows:
```
python run.py COMMAND -c example
```
where `example` is the configuration file `configs/example.yml`. Alternatively, experiments in the config file can be run end-to-end, using:
```
python run.py conduct -c example
```
An example configuration file is provided.

## Currently supported options
### Datasets
* `magnatagatune`: MagnaTagATune (tagging)
* `mtg_jamendo`: MTG Jamendo (tagging)
* `vocalset`: VocalSet (singer recognition, technique recognition)
* `tinysol`: TinySol (instrument recognition, pitch class estimation)
* `beaport`: Beatport (global key detection)

~Many more soon
### Features
* `effnet-discogs`
* `vggish-audioset`
* `msd-musicnn`
* `openl3`
* `neuralfp`
* `clmr-v2`
* `mert-v1-95m-6` / `mert-v1-95m-0-1-2-3` / `mert-v1-95m-4-5-6-7-8` / `mert-v1-95m-9-10-11-12`  (referring to the layers used)
* `maest`

~More soon

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
