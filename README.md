<img src="docs/img/mir_ref_logo.svg" align="left" height="110">

# mir_ref

Representation Evaluation Framework for Music Information Retrieval tasks | [Paper](https://arxiv.org/abs/2312.05994)

`mir_ref` is an open-source library for evaluating audio representations (embeddings or others) on a variety of music-related downstream tasks and datasets. It has two main capabilities:

* Using a config file, you can specify all evaluation experiments you want to run. `mir_ref` will automatically acquire data and conduct the experiments - no coding or data handling needed. Many tasks, datasets, embedding models etc. are ready to use (see supported options section).
* You can easily integrate your own features (e.g. embedding models), datasets, probes, metrics, and audio deformations and use them in `mir_ref` experiments.

`mir_ref` builds upon existing reproducability efforts in musicaudio, including [`mirdata`](https://mirdata.readthedocs.io/en/stable/) for data handling, [`mir_eval`](https://craffel.github.io/mir_eval/) for evaluation metrics, and [`essentia models`](https://essentia.upf.edu/models.html) for pretrained audio models.

## Disclaimer

A first beta release is expected by the end of December, and it includes many fixes and documentation improvements.

## Setup

Clone the repository. Create and activate a python>3.9 environment and install the requirements.

```
cd mir_ref
pip install -r requirements.txt
```

## Running

To run the experiments specified in a config file `configs/example.yml` end-to-end:

```
python run.py conduct -c example
```

This will currently save deformations and features to use them later, but an online option will soon be available.

Alternatively, `mir_ref` is comprised of 4 main functions-commands: `deform`, for generating deformations from a dataset; `extract`, for extracting features; `train`, for training probes; and `evaluate`, for evaluating them. These can be run as follows:

```
python run.py COMMAND -c example
```

`deform` optionally includes the option `n_jobs` for specifying parallelization of deformation computation, and `extract` includes the flag `--no_overwrite` to skip recomputing existing features.

#### Configuration file

An example configuration file is provided. Config files are written in YAML. A list of experiments is expected at the top level, and each experiment contains a task, datasets, features, and probes. For each dataset, a list of deformation scenarios can be specified, following the argument syntax of [audiomentations](https://iver56.github.io/audiomentations/).

## Currently supported options

### Datasets and Tasks

* `magnatagatune`: MagnaTagATune (autotagging)
* `mtg_jamendo`: MTG Jamendo (autotagging)
* `vocalset`: VocalSet (singer_identification, technique_identification)
* `tinysol`: TinySol (instrument_classification, pitch_class_classification)
* `beaport`: Beatport (key_estimation)

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
