"""Wrapper for mirdata datasets, using relevant functions
and adjusting them based on task requirements."""

import mirdata

from mir_ref.datasets.dataset import Dataset


class MirdataDataset(Dataset):
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
        """Dataset wrapper for mirdata dataset.

        Args:
            name (str): Name of the dataset.
            dataset_type (str): Type of the dataset ("mirdata", "custom")
            task_name (str): Name of the task.
            task_type (str): Type of the task.
            data_dir (str, optional): Path to the dataset directory.
                                      Defaults to ./data/{name}/.
            split_type (str, optional): Whether to use "all" or "single" split.
                                        Defaults to "single".
            deformations_cfg (list, optional): List of deformation scenarios.
            features_cfg (list, optional): List of embedding models.
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
        # initialize mirdata dataset
        self.dataset = mirdata.initialize(
            dataset_name=self.name, data_home=self.data_dir
        )

    def download(self):
        self.dataset.download()
        self.dataset.validate(verbose=False)

        # try to load metadata again
        self.load_metadata()

    def download_metadata(self):
        try:
            self.dataset.download(partial_download=["metadata"])
        except ValueError:
            self.dataset.download(partial_download=["annotations"])
        self.dataset.validate(verbose=False)

        # try to load metadata again
        self.load_metadata()

    def preprocess(self):
        """Modifications to the downloaded content of the dataset."""
        return

    def load_track_ids(self):
        if self.task_name == "pitch_class_estimation":
            if self.name == "tinysol":
                track_ids = self.dataset.track_ids
                # only keep track_ids with single pitch annotations
                for track_id in track_ids:
                    if len(self.dataset.track(track_id).pitch) != 1:
                        track_ids.remove(track_id)
        elif self.task_name == "pitch_register_estimation":
            if self.name == "tinysol":
                track_ids = self.dataset.track_ids
                # only keep track_ids with single pitch annotations
                for track_id in track_ids:
                    if len(self.dataset.track(track_id).pitch) != 1:
                        track_ids.remove(track_id)
        elif self.task_name == "key_estimation":
            if self.name == "beatport_key":
                # only keep track_ids with single key annotations
                track_ids = [
                    track_id
                    for track_id in self.dataset.track_ids
                    if len(self.dataset.track(track_id).key) == 1
                    and len(self.dataset.track(track_id).key[0].split(" ")) == 2
                    and "other" not in self.dataset.track(track_id).key[0]
                ]
        else:
            track_ids = self.dataset.track_ids

        self.track_ids = track_ids

    def load_labels(self):
        labels = {}
        for track_id in self.track_ids:
            if (
                self.task_name == "instrument_recognition"
                or self.task_name == "instrument_classification"
            ):
                if self.name == "tinysol":
                    labels[track_id] = self.dataset.track(track_id).instrument_full
            elif self.task_name == "tagging":
                labels[track_id] = self.dataset.track(track_id).tags
            elif self.task_name == "pitch_class_classification":
                pitch = self.dataset.track(track_id).pitch
                labels[track_id] = "".join([c for c in pitch if not c.isdigit()])
            elif self.task_name == "pitch_register_classification":
                pitch = self.dataset.track(track_id).pitch
                labels[track_id] = "".join([c for c in pitch if c.isdigit()])
            elif self.task_name == "key_estimation":
                # map enharmonic keys, always use sharps
                enharm_map = {
                    "Db": "C#",
                    "Eb": "D#",
                    "Gb": "F#",
                    "Ab": "G#",
                    "Bb": "A#",
                    "F#_": "F#",  # ok yes, that's an annotation fix
                }
                key = self.dataset.track(track_id).key[0].strip()
                for pitch_class in enharm_map.keys():
                    if pitch_class in key:
                        key = key.split(" ")
                        key[0] = enharm_map[key[0]]
                        key = " ".join(key)
                        break
                labels[track_id] = key

        self.labels = labels

    def load_audio_paths(self):
        self.audio_paths = {
            t_id: self.dataset.track(t_id).audio_path for t_id in self.track_ids
        }

    # @Dataset.check_metadata_is_loaded
    def get_splits(self, seed=42):
        if self.split_type not in ["all", "single"] or isinstance(
            self.split_type, list
        ):
            raise ValueError(
                "Split type must be 'all', 'single', or "
                + "list of 3 floats adding up to 1."
            )
        # !!!validate metadata exists, and download if not
        # tags can be in metadata or annotations. Try metadata first.
        # try:
        #     self.dataset.download(partial_download=["metadata"])
        # except ValueError:
        #     self.dataset.download(partial_download=["annotations"])
        # self.dataset.validate(verbose=False)

        splits = []
        if self.split_type in ["all", "single"]:
            # check for up to 50 splits
            for i in range(50):
                try:
                    split = self.dataset.get_track_splits(i)
                except TypeError:
                    # if it fails at i=0, there are either no splits or only one split
                    if i == 0:
                        try:
                            split = self.dataset.get_track_splits()
                        except NotImplementedError:
                            # no splits are available, so we need to generate them
                            print(
                                "No official splits found, generating random, stratified "
                                + f"ones with seed {seed}."
                            )
                            splits.append(self.get_stratified_split(seed=seed))
                            break
                    # if it fails at i>0, there are no more splits
                    else:
                        break
                # we need to determine if each split returned is in the format
                # ["train", "validation", "test"], or whether they are actually
                # just folds keyed by integer index.
                try:
                    _, _, _ = split["train"], split["validation"], split["test"]
                    splits.append(split)
                except KeyError:
                    # assume they're folds
                    n_folds = len(split.keys())
                    if n_folds >= 3:
                        # get splits for cross validation, assigning one for
                        # validation and test, and the rest for training
                        for fold_idx in range(n_folds):
                            fold = {}
                            available_folds = list(split.keys())

                            fold["validation"] = split[fold_idx]
                            available_folds.remove(fold_idx)

                            fold["test"] = split[(fold_idx + 1) % n_folds]
                            available_folds.remove((fold_idx + 1) % n_folds)

                            # the rest is train, get lists and flatten
                            fold["train"] = sum(
                                [split[af] for af in available_folds], []
                            )

                            splits.append(fold)
                    else:
                        print(
                            "No official splits found, generating random, stratified "
                            + f"ones with seed {seed}."
                        )
                        splits.append(self.get_stratified_split(seed=seed))
                if i == 0:
                    break

            if self.split_type == "single":
                splits = [splits[0]]

        else:
            # else split_type is a list of sizes, meaning get stratified splits
            splits.append(self.get_stratified_split(sizes=self.split_type, seed=seed))

        return splits
