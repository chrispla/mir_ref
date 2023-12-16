"""Tests for audio deformations."""

import shutil
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

with open("./tests/test_cfg.yml", "r") as f:
    exp_cfg = yaml.safe_load(f)["experiments"][0]

# we need to load and download the dataset before we can test the deformations
from mir_ref.datasets.dataset import get_dataset
from mir_ref.deformations import generate_deformations

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


def test_single_threaded_deformations():
    generate_deformations(
        dataset,
        n_jobs=1,
    )
    assert Path(
        dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=0)
    ).exists()
    assert Path(
        dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=1)
    ).exists()
    assert not Path(
        dataset.audio_paths[first_track_id].replace(".wav", "_deform_2.wav")
    ).exists()
    assert not Path(
        dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=0).replace(
            ".wav", "_deform_0.wav"
        )
    ).exists()

    # delete computed deformations
    shutil.rmtree("tests/data/tinysol/audio_deformed")


def test_multi_threaded_deformations():
    generate_deformations(
        dataset,
        n_jobs=2,
    )
    assert Path(
        dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=0)
    ).exists()
    assert Path(
        dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=1)
    ).exists()
    assert not Path(
        dataset.audio_paths[first_track_id].replace(".wav", "_deform_2.wav")
    ).exists()
    assert not Path(
        dataset.get_deformed_audio_path(track_id=first_track_id, deform_idx=0).replace(
            ".wav", "_deform_0.wav"
        )
    ).exists()

    # delete computed deformations
    shutil.rmtree("tests/data/tinysol/audio_deformed")
