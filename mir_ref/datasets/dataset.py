"""Generic Dataset class and dataset factory.
"""

import os
from pathlib import Path


def get_dataset(dataset_cfg, task_cfg, features_cfg):
    kwargs = {
        "name": dataset_cfg["name"],
        "dataset_type": dataset_cfg["type"],
        "data_dir": dataset_cfg["dir"],
        "split_type": dataset_cfg["split_type"],
        "task_name": task_cfg["name"],
        "task_type": task_cfg["type"],
        "feature_aggregation": task_cfg["feature_aggregation"],
        "deformations_cfg": dataset_cfg["deformations"],
        "features_cfg": features_cfg,
    }
    if dataset_cfg["type"] == "mirdata":
        from mir_ref.datasets.mirdata_datasets import MirdataDataset

        return MirdataDataset(**kwargs)

    elif dataset_cfg["type"] == "custom":
        if dataset_cfg["name"] == "vocalset":
            from mir_ref.datasets.datasets.vocalset import VocalSet

            return VocalSet(**kwargs)
        elif dataset_cfg["name"] in [
            "mtg-jamendo-moodtheme",
            "mtg-jamendo-instruments",
            "mtg-jamendo-genre",
            "mtg-jamendo-top50tags",
        ]:
            from mir_ref.datasets.datasets.mtg_jamendo import MTG_Jamendo

            return MTG_Jamendo(**kwargs)
        elif dataset_cfg["name"] == "magnatagatune":
            from mir_ref.datasets.datasets.magnatagatune import MagnaTagATune

            return MagnaTagATune(**kwargs)
        else:
            raise NotImplementedError(
                f"Custom dataset with name '{dataset_cfg['name']}' does not exist."
            )

    else:
        raise NotImplementedError


class Dataset:
    def __init__(
        self,
        name,
        dataset_type,
        task_name,
        task_type,
        feature_aggregation,
        deformations_cfg,
        features_cfg,
        split_type="single",
        data_dir=None,
    ):
        """Generic Dataset class.

        Args:
            name (str): Name of the dataset.
            dataset_type (str): Type of the dataset ("mirdata", "custom")
            task_name (str): Name of the task.
            task_type (str): Type of the task.
            data_dir (str, optional): Path to the dataset directory.
                                      Defaults to ./data/{name}/.
            split_type (str, optional): Whether to use "all" or "single" split.
                                        Defaults to "single". If list of 3 floats,
                                        use a train, val, test split sizes.
            deformations_cfg (list, optional): List of deformation scenarios.
            features_cfg (list, optional): List of embedding models.
        """
        self.deformations_cfg = deformations_cfg
        self.features_cfg = features_cfg
        self.name = name
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        if data_dir is None:
            self.data_dir = f"./data/{name}/"
        self.split_type = split_type
        self.task_name = task_name
        self.task_type = task_type
        self.feature_aggregation = feature_aggregation
        try:
            self.load_metadata()
        except:
            self.track_ids = None  # list of track_ids
            self.labels = None  # dict of track_id: label
            self.encoded_labels = None  # dict of track_id: encoded_label
            self.audio_paths = None  # dict of track_id: audio_path
            self.common_audio_dir = None  # common directory between all audio paths

    def check_params(self):
        # check validity of provided parameters
        return

    def check_metadata_is_loaded(func):
        """Decorator to check if metadata is loaded."""

        def wrapper(self):
            if any(
                var is None
                for var in [
                    self.track_ids,
                    self.labels,
                    self.encoded_labels,
                    self.audio_paths,
                    self.common_audio_dir,
                ]
            ):
                raise ValueError(
                    "Metadata not loaded. Make sure to run 'dataset.download()'."
                )
            return func(self)

        return wrapper

    def try_to_load_metadata(func):
        """Try to load metadata after a function. Aimed at functions
        that download or preprocess datasets.
        """

        def wrapper(self):
            func(self)
            try:
                self.load_metadata()
            except Exception as e:
                print(e)

        return wrapper

    def download(self):
        # to be overwritten by child class
        return

    def download_metadata(self):
        # to be overwritten by child class
        return

    def preprocess(self):
        # to be overwritten by child class
        return

    def load_track_ids(self):
        # to be overwritten by child class
        return

    def load_audio_paths(self):
        # to be overwritten by child class
        return

    def load_labels(self):
        # to be overwritten by child class
        return

    def load_encoded_labels(self):
        """Return encoded labels given task type and labels."""

        # get lists of track_ids and labels (corresponding indices)
        track_ids = list(self.labels.keys())
        labels_list = list(self.labels.values())

        if self.task_type == "multiclass_classification":
            import keras
            from sklearn.preprocessing import LabelEncoder

            # fit label encoder on all tracks and labels
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels_list)
            # get encoded labels
            encoded_labels_list = self.label_encoder.transform(labels_list)
            encoded_categorical_labels_list = keras.utils.to_categorical(
                encoded_labels_list, num_classes=len(set(labels_list))
            )
            self.encoded_labels = {
                track_ids[i]: encoded_categorical_labels_list[i]
                for i in range(len(track_ids))
            }

        elif self.task_type == "multilabel_classification":
            from sklearn.preprocessing import MultiLabelBinarizer

            # fit label encoder on all tracks and labels
            self.label_encoder = MultiLabelBinarizer()
            self.label_encoder.fit(labels_list)
            # get encoded labels
            encoded_labels_list = self.label_encoder.transform(labels_list)
            self.encoded_labels = {
                track_ids[i]: encoded_labels_list[i] for i in range(len(track_ids))
            }

        else:
            raise NotImplementedError

    def load_common_audio_dir(self):
        """Get the deepest common directory between all audio paths.
        This will later be used as a reference for creating the dir
        structure for deformed audio and embeddings."""

        self.common_audio_dir = str(os.path.commonpath(self.audio_paths.values()))

    def get_deformed_audio_path(self, track_id, deform_idx):
        """Get path of deformed audio based on audio path and deform_idx."""

        audio_path = Path(self.audio_paths[track_id])
        new_filestem = f"{audio_path.stem}_deform_{deform_idx}"
        return str(
            Path(self.data_dir)
            / "audio_deformed"
            / audio_path.relative_to(self.common_audio_dir).with_name(
                f"{new_filestem}{audio_path.suffix}"
            )
        )

    def get_embedding_path(self, track_id, feature):
        """Get path of embedding based on audio path and embedding model."""

        return str(
            Path(self.data_dir)
            / "embeddings"
            / feature
            / Path(self.audio_paths[track_id])
            .relative_to(self.common_audio_dir)
            .with_suffix(f".npy")
        )

    def get_deformed_embedding_path(self, track_id, deform_idx, feature):
        """Get path of deformed embedding based on audio path, embedding model
        and deform_idx."""

        return str(
            Path(self.data_dir)
            / "embeddings"
            / feature
            / Path(self.audio_paths[track_id])
            .relative_to(self.common_audio_dir)
            .with_name(
                f"{Path(self.audio_paths[track_id]).stem}_deform_{deform_idx}.npy"
            )
        )

    def get_stratified_split(self, sizes=(0.8, 0.1, 0.1), seed=42):
        """Helper method to generate a stratified split of the dataset.

        Args:
            sizes (tuple, optional): Sizes of train, validation and test set.
                                     Defaults to (0.8, 0.1, 0.1), must add up to 1.
            seed (int, optional): Random seed. Defaults to 42.
        """
        from sklearn.model_selection import train_test_split

        if sum(sizes) != 1:
            raise ValueError("Sizes must add up to 1.")

        X = self.track_ids
        y = [self.labels[track_id] for track_id in self.track_ids]
        X_train, X_others, y_train, y_others = train_test_split(
            X,
            y,
            test_size=1 - sizes[0],
            random_state=seed,
            stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_others,
            y_others,
            test_size=sizes[2] / (sizes[1] + sizes[2]),
            random_state=seed,
            stratify=y_others,
        )
        return {"train": X_train, "validation": X_val, "test": X_test}

    def load_metadata(self):
        self.load_track_ids()
        self.load_labels()
        self.load_encoded_labels()
        self.load_audio_paths()
        self.load_common_audio_dir()

    def encode_label(self, label):
        return self.label_encoder.transform([label])[0]

    def decode_label(self, encoded_label):
        if self.task_type == "multiclass_classification":
            import numpy as np

            # get index maximum value
            encoded_label = np.argmax(encoded_label)
            return self.label_encoder.inverse_transform([encoded_label])[0]
        return self.label_encoder.inverse_transform([encoded_label])[0]
