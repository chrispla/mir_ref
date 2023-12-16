"""Tests for dataset objects."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from mir_ref.datasets.dataset import get_dataset

# load configuration file, used in all tests
# it uses the tinysol dataset, implemented with mirdata
with open("./tests/test_cfg.yml", "r") as f:
    exp_cfg = yaml.safe_load(f)["experiments"][0]

dataset = get_dataset(
    exp_cfg["datasets"][0],
    exp_cfg["task"],
    exp_cfg["features"],
)


def test_download_metadata():
    dataset.download_metadata()
    assert Path("./tests/data/tinysol/annotation").is_dir()
    assert Path("./tests/data/tinysol/annotation/TinySOL_metadata.csv").exists()


def test_load_metadata():
    dataset.load_metadata()
    assert (
        dataset.common_audio_dir == "tests/data/tinysol/audio"
        or dataset.common_audio_dir == "tests/data/tinysol/audio/"
    )
    assert dataset.track_ids[0] == "BTb-ord-F#1-pp-N-N"
    assert dataset.labels["BTb-ord-F#1-pp-N-N"] == "Bass Tuba"
    assert len(set(dataset.labels.values())) == 14
    categorical_encoded_label = list(np.zeros((14)).astype(np.float32))
    categorical_encoded_label[2] = 1.0
    assert np.array_equal(
        dataset.encoded_labels["BTb-ord-F#1-pp-N-N"], categorical_encoded_label
    )
    assert (
        dataset.audio_paths["BTb-ord-F#1-pp-N-N"]
        == "tests/data/tinysol/audio/Brass/Bass_Tuba/ordinario/"
        + "BTb-ord-F#1-pp-N-N.wav"
    )


def test_extended_metadata():
    dataset.load_metadata()

    deformed_audio_path = dataset.get_deformed_audio_path("BTb-ord-F#1-pp-N-N", 0)
    assert deformed_audio_path == (
        "tests/data/tinysol/audio_deformed/Brass/Bass_Tuba/ordinario/"
        + "BTb-ord-F#1-pp-N-N_deform_0.wav"
    )
    emb_path = dataset.get_embedding_path("BTb-ord-F#1-pp-N-N", "vggish-audioset")
    assert emb_path == (
        "tests/data/tinysol/embeddings/vggish-audioset/Brass/Bass_Tuba/ordinario/"
        + "BTb-ord-F#1-pp-N-N.npy"
    )
    deformed_emb_path = dataset.get_deformed_embedding_path(
        "BTb-ord-F#1-pp-N-N", 0, "vggish-audioset"
    )
    assert deformed_emb_path == (
        "tests/data/tinysol/embeddings/vggish-audioset/Brass/Bass_Tuba/ordinario/"
        + "BTb-ord-F#1-pp-N-N_deform_0.npy"
    )


test_load_metadata()
