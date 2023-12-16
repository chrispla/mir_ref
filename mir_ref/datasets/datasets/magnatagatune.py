import csv
import os.path
import zipfile
from pathlib import Path

import numpy as np
import wget
from tqdm import tqdm

from mir_ref.datasets.dataset import Dataset


class MagnaTagATune(Dataset):
    def __init__(
        self,
        name,
        dataset_type,
        data_dir,
        split_type,
        task_name,
        task_type,
        feature_aggregation,
        deformations_cfg,
        features_cfg,
    ):
        """Dataset wrapper for MagnaTagATune."""
        super().__init__(
            name=name,
            dataset_type=dataset_type,
            data_dir=data_dir,
            split_type=split_type,
            task_name=task_name,
            task_type=task_type,
            feature_aggregation=feature_aggregation,
            deformations_cfg=deformations_cfg,
            features_cfg=features_cfg,
        )

    @Dataset.try_to_load_metadata
    def download(self):
        # make data dir if it doesn't exist or if it exists but is empty
        if os.path.exists(os.path.join(self.data_dir, "audio")) and (
            len(os.listdir(os.path.join(self.data_dir, "audio"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Dataset '{self.name}' already exists in '{self.data_dir}'."
                + "Skipping audio download.",
                stacklevel=2,
            )
            self.download_metadata()
            return
        (Path(self.data_dir) / "audio").mkdir(parents=True, exist_ok=True)

        print(f"Downloading MagnaTagATune to {self.data_dir}...")
        for i in tqdm(["001", "002", "003"]):
            wget.download(
                url=f"https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.{i}",
                out=os.path.join(self.data_dir, "audio/"),
            )

        archive_dir = os.path.join(self.data_dir, "audio")

        # Combine the split archive files into a single file
        with open(os.path.join(archive_dir, "mp3.zip"), "wb") as f:
            for i in ["001", "002", "003"]:
                with open(
                    os.path.join(archive_dir, f"mp3.zip.{i}"),
                    "rb",
                ) as part:
                    f.write(part.read())

        # Extract the contents of the archive
        with zipfile.ZipFile(os.path.join(archive_dir, "mp3.zip"), "r") as zip_ref:
            zip_ref.extractall()

        # Remove zips
        for i in ["", ".001", ".002", ".003"]:
            os.remove(os.path.join(archive_dir, f"mp3.zip{i}"))

        self.download_metadata()

    @Dataset.try_to_load_metadata
    def download_metadata(self):
        if os.path.exists(os.path.join(self.data_dir, "metadata")) and (
            len(os.listdir(os.path.join(self.data_dir, "metadata"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Metadata for dataset '{self.name}' already exists in '{self.data_dir}'."
                + "Skipping metadata download.",
                stacklevel=2,
            )
            return
        (Path(self.data_dir) / "metadata").mkdir(parents=True, exist_ok=True)

        urls = [
            # annotations
            "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv",
            # train, validation, and test splits from Won et al. 2020
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy",
        ]
        for url in urls:
            wget.download(
                url=url,
                out=os.path.join(self.data_dir, "metadata/"),
            )

    def load_track_ids(self):
        with open(
            os.path.join(self.data_dir, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.track_ids = [line[0] for line in annotations]
            # manually remove some corrupt files
            self.track_ids.remove("35644")
            self.track_ids.remove("55753")
            self.track_ids.remove("57881")

    def load_labels(self):
        # get the list of top 50 tags used in Minz Won et al. 2020
        tags = np.load(os.path.join(self.data_dir, "metadata", "tags.npy"))

        with open(
            os.path.join(self.data_dir, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            annotations_header = next(annotations)
            self.labels = {
                line[0]: [
                    annotations_header[j]
                    for j in range(1, len(line) - 1)
                    # only add the tag if it's in the tags list
                    if line[j] == "1" and annotations_header[j] in tags
                ]
                for line in annotations
                # this is a slow way to do it, temporary fix for
                # some corrupt mp3s
                if line[0] in self.track_ids
            }

    def load_audio_paths(self):
        with open(
            os.path.join(self.data_dir, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.audio_paths = {
                line[0]: os.path.join(self.data_dir, "audio", line[-1])
                for line in annotations
                # this is a slow way to do it, temporary fix for
                # some corrupt mp3s
                if line[0] in self.track_ids
            }

    @Dataset.check_metadata_is_loaded
    def get_splits(self):
        # get inverse dictionary to get track id from audio path
        rel_path_to_track_id = {
            (Path(v).parent.name + "/" + Path(v).name): k
            for k, v in self.audio_paths.items()
        }

        split = {}
        for set_filename, set_targetname in zip(
            ["train", "valid", "test"], ["train", "validation", "test"]
        ):
            relative_paths = np.load(
                os.path.join(self.data_dir, "metadata", f"{set_filename}.npy")
            )
            # get track_ids by getting the full path and using the inv dict
            split[set_targetname] = [
                rel_path_to_track_id[path.split("\t")[1]] for path in relative_paths
            ]

        if self.split_type not in ["all", "single"]:
            raise NotImplementedError(f"Split type '{self.split_type}' does not exist.")

        return [split]
