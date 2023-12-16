import glob
import os
import subprocess
from pathlib import Path

import wget

from mir_ref.datasets.dataset import Dataset


class VocalSet(Dataset):
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
        """Dataset wrapper for VocalSet dataset."""
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
        if Path(self.data_dir).exists():
            import warnings

            warnings.warn(
                (
                    f"Dataset {self.name} already exists at {self.data_dir}."
                    + " Skipping download."
                )
            )
            self.preprocess()
            return
        # make data dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=False)
        zenodo_url = "https://zenodo.org/record/1203819/files/VocalSet11.zip"
        print(f"Downloading VocalSet to {self.data_dir}...")
        wget.download(zenodo_url, self.data_dir)
        # extract zip
        subprocess.run(
            [
                "unzip",
                "-q",
                "-d",
                str(Path(self.data_dir)),
                str(Path(self.data_dir) / "VocalSet11.zip"),
            ]
        )
        # remove zip
        subprocess.run(["rm", str(Path(self.data_dir) / "VocalSet11.zip")])

        # preprocess
        self.preprocess()

    def preprocess(self):
        # need to make some corrections to filenames and delete some duplicates
        # Thanks to the MARBLE authors for the list of corrections and dups:
        # https://github.com/a43992899/MARBLE-Benchmark/blob/main/benchmark/tasks/VocalSet/preprocess.py

        if not (
            Path(self.data_dir)
            / "FULL"
            / "**"
            / "*"
            / "vibrato/f2_scales_vibrato_a(1).wav"
        ).exists():
            # dataset is probably already preprocessed
            return

        file_delete = [
            "vibrato/f2_scales_vibrato_a(1).wav",
            "vibrato/caro_vibrato.wav",
            "vibrato/dona_vibrato.wav",
            "vibrato/row_vibrato.wav",
            "vibrado/slow_vibrato_arps.wav",
        ]

        filepaths_to_delete = [
            glob.glob(str(Path(self.data_dir) / "FULL" / "**" / "*" / f))
            for f in file_delete
        ]
        # flatten list of lists
        filepaths_to_delete = [
            item for sublist in filepaths_to_delete for item in sublist
        ]
        # delete files
        for f in filepaths_to_delete:
            os.remove(f)

        # thanks black formatter for this beauty
        name_correction = [
            ("/lip_trill/lip_trill_arps.wav", "/lip_trill/f2_lip_trill_arps.wav"),
            ("/lip_trill/scales_lip_trill.wav", "/lip_trill/m3_scales_lip_trill.wav"),
            (
                "/straight/arpeggios_straight_a.wav",
                "/straight/f4_arpeggios_straight_a.wav",
            ),
            (
                "/straight/arpeggios_straight_e.wav",
                "/straight/f4_arpeggios_straight_e.wav",
            ),
            (
                "/straight/arpeggios_straight_i.wav",
                "/straight/f4_arpeggios_straight_i.wav",
            ),
            (
                "/straight/arpeggios_straight_o.wav",
                "/straight/f4_arpeggios_straight_o.wav",
            ),
            (
                "/straight/arpeggios_straight_u.wav",
                "/straight/f4_arpeggios_straight_u.wav",
            ),
            ("/straight/row_straight.wav", "/straight/m8_row_straight.wav"),
            ("/straight/scales_straight_a.wav", "/straight/f4_scales_straight_a.wav"),
            ("/straight/scales_straight_e.wav", "/straight/f4_scales_straight_e.wav"),
            ("/straight/scales_straight_i.wav", "/straight/f4_scales_straight_i.wav"),
            ("/straight/scales_straight_o.wav", "/straight/f4_scales_straight_o.wav"),
            ("/straight/scales_straight_u.wav", "/straight/f4_scales_straight_u.wav"),
            ("/vocal_fry/scales_vocal_fry.wav", "/vocal_fry/f2_scales_vocal_fry.wav"),
            (
                "/fast_forte/arps_fast_piano_c.wav",
                "/fast_forte/f9_arps_fast_piano_c.wav",
            ),
            (
                "/fast_piano/fast_piano_arps_f.wav",
                "/fast_piano/f2_fast_piano_arps_f.wav",
            ),
            (
                "/fast_piano/arps_c_fast_piano.wav",
                "/fast_piano/m3_arps_c_fast_piano.wav",
            ),
            (
                "/fast_piano/scales_fast_piano_f.wav",
                "/fast_piano/f3_scales_fast_piano_f.wav",
            ),
            (
                "/fast_piano/scales_c_fast_piano_a.wav",
                "/fast_piano/m10_scales_c_fast_piano_a.wav",
            ),
            (
                "/fast_piano/scales_c_fast_piano_e.wav",
                "/fast_piano/m10_scales_c_fast_piano_e.wav",
            ),
            (
                "/fast_piano/scales_c_fast_piano_i.wav",
                "/fast_piano/m10_scales_c_fast_piano_i.wav",
            ),
            (
                "/fast_piano/scales_c_fast_piano_o.wav",
                "/fast_piano/m10_scales_c_fast_piano_o.wav",
            ),
            (
                "/fast_piano/scales_c_fast_piano_u.wav",
                "/fast_piano/m10_scales_c_fast_piano_u.wav",
            ),
            (
                "/fast_piano/scales_f_fast_piano_a.wav",
                "/fast_piano/m10_scales_f_fast_piano_a.wav",
            ),
            (
                "/fast_piano/scales_f_fast_piano_e.wav",
                "/fast_piano/m10_scales_f_fast_piano_e.wav",
            ),
            (
                "/fast_piano/scales_f_fast_piano_i.wav",
                "/fast_piano/m10_scales_f_fast_piano_i.wav",
            ),
            (
                "/fast_piano/scales_f_fast_piano_o.wav",
                "/fast_piano/m10_scales_f_fast_piano_o.wav",
            ),
            (
                "/fast_piano/scales_f_fast_piano_u.wav",
                "/fast_piano/m10_scales_f_fast_piano_u.wav",
            ),
        ]

        for old, new in name_correction:
            old_matches = glob.glob(
                str(Path(self.data_dir) / "FULL" / "**" / "*" / old[1:])
            )
            target_filepaths = [f.replace(old, new) for f in old_matches]
            # rename
            for old, new in zip(old_matches, target_filepaths):
                os.rename(old, new)

    @Dataset.try_to_load_metadata
    def download_metadata(self):
        # not possible to download only metadata for vocalset
        self.download()

    def load_track_ids(self):
        # the names of the tracks can be used as a unique identifier, as they contain
        # singer, technique, and take index information.

        if self.task_name == "singer_identification":
            self.track_ids = [
                Path(path).stem
                for path in glob.glob(
                    str(Path(self.data_dir) / "FULL/**/*.wav"), recursive=True
                )
            ]
        elif self.task_name == "technique_identification":
            # use the 10 techniques used in the original paper
            techniques = [
                "vibrato",
                "straight",
                "belt",
                "breathy",
                "lip_trill",
                "spoken",
                "inhaled",
                "trill",
                "trillo",
                "vocal_fry",
            ]
            # only add track_ids that contain one of the techniques
            self.track_ids = [
                Path(path).stem
                for path in glob.glob(
                    str(Path(self.data_dir) / "FULL/**/*.wav"), recursive=True
                )
                if any(technique in Path(path).stem for technique in techniques)
            ]

    def load_labels(self):
        if self.task_name == "singer_identification":
            self.labels = {
                Path(path).stem: (Path(path).stem)[:2]
                if ((Path(path).stem)[:3] != "m10" and (Path(path).stem)[:3] != "m11")
                else (Path(path).stem)[:3]
                for path in glob.glob(
                    str(Path(self.data_dir) / "FULL/**/*.wav"), recursive=True
                )
            }
        elif self.task_name == "technique_identification":
            # use the 10 techniques used in the original paper
            techniques = [
                "vibrato",
                "straight",
                "belt",
                "breathy",
                "lip_trill",
                "spoken",
                "inhaled",
                "trill",
                "trillo",
                "vocal_fry",
            ]
            labels = {}
            for track_id in self.track_ids:
                for technique in techniques:
                    if technique in track_id:
                        labels[track_id] = technique
            self.labels = labels
        else:
            raise NotImplementedError(
                f"Task '{self.task_name}' not available for this dataset."
            )

    def load_audio_paths(self):
        audio_paths_list = [
            path
            for path in glob.glob(
                str(Path(self.data_dir) / "FULL/**/*.wav"), recursive=True
            )
        ]
        self.audio_paths = {Path(path).stem: path for path in audio_paths_list}

    @Dataset.check_metadata_is_loaded
    def get_splits(self):
        if self.task_name == "singer_identification":
            # no official splits are available, get stratified one
            return [super().get_stratified_split()]
        elif self.task_name == "technique_identification":
            train_singers = [
                "f1",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f9",
                "m1",
                "m2",
                "m4",
                "m6",
                "m7",
                "m8",
                "m9",
            ]
            # the original paper does not have a validation set, so we steal f9,
            # m9, and m11 from train
            val_singers = ["f9", "m9", "m11"]
            test_singer = ["f2", "f8", "m3", "m5", "m10"]

            split = {}
            split["train"] = [
                track_id for track_id in self.track_ids if track_id[:2] in train_singers
            ]
            split["validation"] = [
                track_id for track_id in self.track_ids if track_id[:2] in val_singers
            ]
            split["test"] = [
                track_id for track_id in self.track_ids if track_id[:2] in test_singer
            ]
            return [split]
        else:
            raise NotImplementedError(
                f"Task '{self.task_name}' not available for this dataset."
            )
