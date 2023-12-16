import csv
import hashlib
import os.path
import sys
import tarfile
from pathlib import Path

import tqdm
import wget

from mir_ref.datasets.dataset import Dataset


class MTG_Jamendo(Dataset):
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
        """Dataset wrapper for MTG Jamendo (sub)dataset(s). Each subset
        (moodtheme, instrument, top50tags, genre) is going to be treated
        as a separate dataset, meaning a separate data_dir needs to be
        specified for each one. This helps disambiguate versioning of
        the deformations and experiments. Since the methods are shared,
        however, a single class is used for all of them.
        """
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
        # download with mtg_jamendo download helper
        if self.name == "mtg-jamendo-moodtheme":
            # only moodtheme has a separate download target
            download_jamendo(
                dataset="autotagging_moodtheme",
                data_type="audio-low",
                download_from="mtg-fast",
                output_dir=os.path.join(self.data_dir, "audio/"),
                unpack_tars=True,
                remove_tars=True,
            )
        elif (
            self.name == "mtg-jamendo-instrument"
            or self.name == "mtg-jamendo-genre"
            or self.name == "mtg-jamendo-top50tags"
        ):
            # the whole dataset needs to be downloaded as no subset-specific
            # download targets are available
            download_jamendo(
                dataset="raw_30s",
                data_type="audio-low",
                download_from="mtg-fast",
                output_dir=os.path.join(self.data_dir, "audio/"),
                unpack_tars=True,
                remove_tars=True,
            )
            # optionally I'd delete unneeded tracks here...
        else:
            raise NotImplementedError(f"Dataset '{self.name}' does not exist.")
        self.download_metadata()

    @Dataset.try_to_load_metadata
    def download_metadata(self):
        # download from github link
        url = (
            "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/"
            + "master/data/autotagging_"
            + (self.name).split("-")[2]
            + ".tsv"
        )
        (Path(self.data_dir) / "metadata").mkdir(parents=True, exist_ok=True)
        wget.download(url, out=os.path.join(self.data_dir, "metadata/"))

    def load_track_ids(self):
        # open the corresponding file using the subset name in self.name
        with open(
            os.path.join(
                self.data_dir,
                "metadata/",
                f"autotagging_{(self.name).split('-')[2]}.tsv",
            ),
            "r",
        ) as f:
            metadata = f.readlines()
            self.track_ids = [
                line.split("\t")[0].strip() for line in metadata[1:] if line
            ]

    def load_labels(self):
        with open(
            os.path.join(
                self.data_dir,
                "metadata/",
                f"autotagging_{(self.name).split('-')[2]}.tsv",
            ),
            "r",
        ) as f:
            metadata = f.readlines()
            # for each line, get key (track_id) and all stripped entries after
            # column 5 (list of labels)
            self.labels = {
                line.split("\t")[0].strip(): [
                    tag.strip() for tag in line.split("\t")[5:]
                ]
                for line in metadata[1:]
            }

    def load_audio_paths(self):
        with open(
            os.path.join(
                self.data_dir,
                "metadata/",
                f"autotagging_{(self.name).split('-')[2]}.tsv",
            ),
            "r",
        ) as f:
            metadata = f.readlines()
            # the lowres version we downloaded contains "low" before the extension
            self.audio_paths = {
                line.split("\t")[0].strip(): os.path.join(
                    self.data_dir,
                    "audio",
                    (line.split("\t")[3].strip())[:-4] + ".low.mp3",
                )
                for line in metadata[1:]
                if line
            }

    @Dataset.check_metadata_is_loaded
    def get_splits(self):
        subset = (self.name).split("-")[2]
        splits_url_dir = (
            "https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/splits/"
        )

        if not os.path.exists(os.path.join(self.data_dir, "splits/")):
            # there are 5 splits for each subset
            for i in range(5):
                (Path(self.data_dir) / "metadata" / "splits").mkdir(
                    parents=True, exist_ok=True
                )
                for split in ["train", "validation", "test"]:
                    url = f"{splits_url_dir}split-{i}/autotagging_{subset}-{split}.tsv"
                    wget.download(
                        url,
                        out=os.path.join(
                            self.data_dir, "metadata", "splits", f"split-{i}"
                        ),
                    )
        splits = []
        for i in range(5):
            splits.append({})
            for split in ["train", "validation", "test"]:
                with open(
                    os.path.join(
                        self.data_dir,
                        "metadata",
                        "splits",
                        f"split-{i}",
                        f"autotagging_{subset}-{split}.tsv",
                    ),
                    "r",
                ) as f:
                    split_metadata = f.readlines()
                    self.splits[i][split] = [
                        line.split("\t")[0].strip()
                        for line in split_metadata[1:]
                        if line
                    ]
        if self.split_type == "single":
            return [splits[0]]
        elif self.split_type == "all":
            return splits
        else:
            raise NotImplementedError(f"Split type '{self.split_type}' does not exist.")


"""Code to download MTG Jamendo.
Source: https://github.com/MTG/mtg-jamendo-dataset
"""

download_from_names = {"gdrive": "GDrive", "mtg": "MTG", "mtg-fast": "MTG Fast mirror"}


def compute_sha256(filename):
    with open(filename, "rb") as f:
        contents = f.read()
        checksum = hashlib.sha256(contents).hexdigest()
        return checksum


def download_jamendo(
    dataset, data_type, download_from, output_dir, unpack_tars, remove_tars
):
    if not os.path.exists(output_dir):
        print("Output directory {} does not exist".format(output_dir), file=sys.stderr)
        return

    if download_from not in download_from_names:
        print(
            "Unknown --from argument, choices are {}".format(
                list(download_from_names.keys())
            ),
            file=sys.stderr,
        )
        return

    print("Downloading %s from %s" % (dataset, download_from_names[download_from]))
    # download checksums
    file_sha256_tars_url = (
        "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/download/"
        + f"{dataset}_{data_type}_sha256_tars.txt"
    )
    wget.download(file_sha256_tars_url, out=output_dir)
    file_sha256_tracks_url = (
        "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/download/"
        + f"{dataset}_{data_type}_sha256_tracks.txt"
    )
    wget.download(file_sha256_tracks_url, out=output_dir)

    file_sha256_tars = os.path.join(
        output_dir, dataset + "_" + data_type + "_sha256_tars.txt"
    )
    file_sha256_tracks = os.path.join(
        output_dir, dataset + "_" + data_type + "_sha256_tracks.txt"
    )

    # Read checksum values for tars and files inside.
    with open(file_sha256_tars) as f:
        sha256_tars = dict([(row[1], row[0]) for row in csv.reader(f, delimiter=" ")])

    with open(file_sha256_tracks) as f:
        sha256_tracks = dict([(row[1], row[0]) for row in csv.reader(f, delimiter=" ")])

    # Filenames to download.
    ids = sha256_tars.keys()

    removed = []
    for filename in tqdm(ids, total=len(ids)):
        output = os.path.join(output_dir, filename)
        # print(output)
        # print(filename)

        # Download from Google Drive.
        if os.path.exists(output):
            print("Skipping %s (file already exists)" % output)
            continue

        elif download_from == "mtg":
            url = (
                "https://essentia.upf.edu/documentation/datasets/mtg-jamendo/"
                "%s/%s/%s" % (dataset, data_type, filename)
            )
            # print("From:", url)
            # print("To:", output)
            wget.download(url, out=output)

        elif download_from == "mtg-fast":
            url = "https://cdn.freesound.org/mtg-jamendo/" "%s/%s/%s" % (
                dataset,
                data_type,
                filename,
            )
            # print("From:", url)
            # print("To:", output)
            wget.download(url, out=output)

        # Validate the checksum.
        if compute_sha256(output) != sha256_tars[filename]:
            print(
                "%s does not match the checksum, removing the file" % output,
                file=sys.stderr,
            )
            removed.append(filename)
            os.remove(output)
        # else:
        #     print("%s checksum OK" % filename)

    if removed:
        print("Missing files:", " ".join(removed))
        print("Re-run the script again")
        return

    print("Download complete")

    if unpack_tars:
        print("Unpacking tar archives")

        tracks_checked = []
        for filename in tqdm(ids, total=len(ids)):
            output = os.path.join(output_dir, filename)
            print("Unpacking", output)
            tar = tarfile.open(output)
            tracks = tar.getnames()[1:]  # The first element is folder name.
            tar.extractall(path=output_dir)
            tar.close()

            # Validate checksums for all unpacked files
            for track in tracks:
                trackname = os.path.join(output_dir, track)
                if compute_sha256(trackname) != sha256_tracks[track]:
                    print("%s does not match the checksum" % trackname, file=sys.stderr)
                    raise Exception("Corrupt file in the dataset: %s" % trackname)

            # print("%s track checksums OK" % filename)
            tracks_checked += tracks

            if remove_tars:
                os.remove(output)

        # Check if any tracks are missing in the unpacked archives.
        if set(tracks_checked) != set(sha256_tracks.keys()):
            raise Exception(
                "Unpacked data contains tracks not present in the checksum files"
            )

        print("Unpacking complete")

    # delete checksum files
    os.remove(file_sha256_tars)
    os.remove(file_sha256_tracks)
