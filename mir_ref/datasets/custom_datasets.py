"""Implement your own datasets here. Provide custom implementations
for the given methods, without changing any of the given method
and input/output names.
"""

from mir_ref.datasets.dataset import Dataset


class CustomDataset0(Dataset):
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
        """Custom Dataset class.

        Args:
            name (str): Name of the dataset.
            dataset_type (str): Type of the dataset ("mirdata", "custom")
            data_dir (str): Path to the dataset directory.
            split_type (str): split_type (str, optional): Whether to use "all" or "single" split.
                              Defaults to "single". If list of 3 floats,
                              use a train, val, test split sizes.
            task_name (str): Name of the task.
            task_type (str): Type of the task. ("multiclass_classification",
                             "multilabel_classification", "regression")
            feature_aggregation (str): Type of embedding aggregation ("mean", None)
            deformations_cfg (dict): Deformations config.
            features_cfg (dict): Embedding models config.
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
        pass

    @Dataset.try_to_load_metadata
    def download_metadata(self):
        # optional method
        pass

    def load_track_ids(self):
        # return list of track_ids
        pass

    def load_audio_paths(self):
        # return dict of track_id: audio_path
        pass

    @Dataset.check_metadata_is_loaded
    def get_splits(self):
        # return list of dicts with {train: [track_id_0, ...],
        # test: [", ...], val: [", ...]}
        pass
