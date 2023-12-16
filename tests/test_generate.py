"""Tests embedding inference."""

import shutil
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

with open("./tests/test_cfg.yml", "r") as f:
    exp_cfg = yaml.safe_load(f)["experiments"][0]

# we need to load and download the dataset before we can test the deformations
from mir_ref.datasets.dataset import get_dataset
from mir_ref.features.feature_extraction import generate_embeddings

dataset = get_dataset(
    dataset_cfg=exp_cfg["datasets"][0],
    task_cfg=exp_cfg["task"],
    features_cfg=exp_cfg["features"],
)

dataset.download()
dataset.preprocess()
dataset.load_metadata()

# keep only a few track_ids for testing
# this also tests if everything is correctly anchored to the dataset object
# and its track_ids
# unfortunately we can't currently download only a few tracks from an mirdata dataset
dataset.track_ids = dataset.track_ids[:5]
first_track_id = dataset.track_ids[0]


def test_models():
    dataset.deformations_cfg = None
    for model_name in exp_cfg["features"]:
        generate_embeddings(
            dataset,
            model_name=model_name,
        )

        assert Path(
            dataset.get_embedding_path(track_id=first_track_id, feature=model_name)
        ).exists()

    # delete computed embeddings
    shutil.rmtree("tests/data/tinysol/embeddings")


def test_models_deformed_audio():
    dataset.deformations_cfg = exp_cfg["deformations"]
    # we need to first compute the deformations
    from mir_ref.deformations import generate_deformations

    generate_deformations(
        dataset,
        n_jobs=2,
    )

    for model_name in exp_cfg["features"]:
        generate_embeddings(
            dataset,
            model_name=model_name,
        )
        assert Path(
            dataset.get_embedding_path(track_id=first_track_id, feature=model_name)
        ).exists()
        assert Path(
            dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=0)
        )
        assert Path(
            dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=1)
        )

    # delete computed deformations
    shutil.rmtree("tests/data/tinysol/audio_deformed")

    # delete computed embeddings
    shutil.rmtree("tests/data/tinysol/embeddings")


# test_models()
