"""Train downstream models.
"""

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from sklearn.model_selection import ParameterGrid
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from mir_ref.dataloaders import DataGenerator
from mir_ref.datasets.dataset import get_dataset
from mir_ref.probes.probe_builder import get_model
from mir_ref.utils import load_config


def train(cfg_path, run_id=None):
    """Make a grid of all combinations of dataset, embedding models,
    downstream models and splits and call training for each.

    Args:
        cfg_path (str): Path to config file.
        run_id (str, optional): Experiment ID, timestamp if not specified.
    """

    KERAS_VERSION = int(keras.__version__.split(".")[0])

    cfg = load_config(cfg_path)

    if run_id is None:
        import datetime

        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for exp_cfg in cfg["experiments"]:
        run_params = {
            "dataset_cfg": exp_cfg["datasets"],
            "feature": exp_cfg["features"],
            "model_cfg": exp_cfg["probes"],
        }

        # create grid from parameters
        grid = ParameterGrid(run_params)

        for params in grid:
            # get index of downstream model for naming-logging
            model_idx = exp_cfg["probes"].index(params["model_cfg"])
            # get dataset object
            dataset = get_dataset(
                dataset_cfg=params["dataset_cfg"],
                task_cfg=exp_cfg["task"],
                features_cfg=exp_cfg["features"],
            )
            dataset.download()
            # !!!the following only works if there's one experiment per config
            if dataset.task_type == "multiclass_classification":
                try:
                    for metric in metrics:
                        if KERAS_VERSION == 3:
                            metric.reset_state()
                        else:
                            metric.reset_states()
                except UnboundLocalError:
                    metrics = [
                        keras.metrics.CategoricalAccuracy(),
                        keras.metrics.Precision(),
                        keras.metrics.Recall(),
                        keras.metrics.AUC(),
                    ]
            elif dataset.task_type == "multilabel_classification":
                try:
                    for metric in metrics:
                        if KERAS_VERSION == 3:
                            metric.reset_state()
                        else:
                            metric.reset_states()
                except UnboundLocalError:
                    metrics = [
                        keras.metrics.Precision(),
                        keras.metrics.Recall(),
                        keras.metrics.AUC(curve="ROC"),
                        keras.metrics.AUC(curve="PR"),
                    ]
            # run task for every split
            for split_idx in range(len(dataset.get_splits())):
                print(
                    Fore.GREEN
                    + f"Task: {dataset.task_name}\n"
                    + f"└── Dataset: {dataset.name}\n"
                    + f"    └── Embeddings: {params['feature']}\n"
                    + f"        └── Model: {model_idx}\n"
                    + f"            └── Split: {split_idx}",
                    Style.RESET_ALL,
                )
                train_probe(
                    run_id=run_id,
                    dataset=dataset,
                    model_cfg=params["model_cfg"],
                    model_idx=model_idx,
                    feature=params["feature"],
                    split_idx=split_idx,
                    metrics=metrics,
                )


def train_probe(run_id, dataset, model_cfg, model_idx, feature, split_idx, metrics):
    """Train a single model per split given parameters.

    Args:
        run_id (str): ID of the current run, defaults to timestamp.
        dataset (Dataset): Dataset object.
        model_cfg (dict): Downstream model config.
        model_idx (int): Index of the downstream model in the list of models.
        feature (str): Name of the embedding model.
        split_idx (int): Index of the split in the list of splits.
        metrics (list): List of metrics to use for training.
    """

    KERAS_VERSION = int(keras.__version__.split(".")[0])

    split = dataset.get_splits()[split_idx]
    n_classes = len(dataset.encoded_labels[dataset.track_ids[0]])

    if model_cfg["emb_shape"] == "infer":
        # get embedding shape from the first embedding
        emb_shape = np.load(
            dataset.get_embedding_path(
                feature=feature,
                track_id=split["train"][0],
            )
        ).shape
    elif isinstance(model_cfg["emb_shape"], int):
        emb_shape = model_cfg["emb_shape"]
    elif isinstance(model_cfg["emb_shape"], str):
        raise ValueError(f"{model_cfg['emb_shape']} not implemented.")

    model = get_model(model_cfg=model_cfg, dim=emb_shape, n_classes=n_classes)
    model.summary()

    tr_gen = DataGenerator(
        ids_list=split["train"],
        labels_dict=dataset.encoded_labels,
        paths_dict={
            t_id: dataset.get_embedding_path(feature=feature, track_id=t_id)
            for t_id in split["train"]
        },
        batch_size=model_cfg["batch_size"],
        dim=emb_shape,
        n_classes=n_classes,
        shuffle=True,
    )
    val_gen = DataGenerator(
        ids_list=split["validation"],
        labels_dict=dataset.encoded_labels,
        paths_dict={
            t_id: dataset.get_embedding_path(feature=feature, track_id=t_id)
            for t_id in split["validation"]
        },
        batch_size=1,
        dim=emb_shape,
        n_classes=n_classes,
        shuffle=True,
    )

    # dir for tensorboard logs and weights for this run
    run_dir = (
        Path(run_id) / dataset.task_name / f"{dataset.name}_{feature}_model-{model_idx}"
    )

    # if the dir already exists, change run dir for duplicates
    if (Path("./logs") / run_dir).exists():
        i = 1
        while (Path("./logs") / run_dir).exists():
            run_dir = (
                Path(run_id)
                / dataset.task_name
                / f"{dataset.name}_{feature}_model-{model_idx} ({i})"
            )
            i += 1
        import warnings

        # raise warning about existing experiment
        warnings.warn(
            f"Model in '{dataset.name}_{feature}_model-{model_idx}' already exists. "
            f"Renaming new run to '{run_dir}'",
            stacklevel=2,
        )

    # create dir for this run if it doesn't exist
    if not (Path("./logs") / run_dir).exists():
        (Path("./logs") / run_dir).mkdir(parents=True)

    # save model config in run dir
    with open(Path("./logs") / run_dir / "model_config.yml", "w+") as f:
        yaml.dump(model_cfg, f)

    if KERAS_VERSION == 3:
        model_extension = "keras"
    else:
        model_extension = "h5"

    if dataset.task_type == "multiclass_classification":
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            save_weights_only=False,
            filepath=str(Path("./logs") / run_dir / f"weights.{model_extension}"),
            save_best_only=True,
            monitor="val_categorical_accuracy",
            mode="max",
        )
    elif dataset.task_type == "multilabel_classification":
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            save_weights_only=False,
            filepath=str(Path("./logs") / run_dir / "weights.h5"),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        )
    callbacks = [
        EarlyStopping(patience=model_cfg["patience"]),
        TensorBoard(log_dir=str(Path("./logs") / run_dir)),
        checkpoint_callback,
    ]

    # make sure all metric and callback states are reset
    for callback in callbacks:
        if hasattr(callback, "reset_state"):
            callback.reset_state()

    # loss and optimizer
    if dataset.task_type == "multiclass_classification":
        loss = losses.CategoricalCrossentropy(from_logits=False)
    elif dataset.task_type == "multilabel_classification":
        loss = losses.BinaryCrossentropy()
    else:
        raise ValueError(f"Task type '{dataset.task_type}' not implemented.")

    if model_cfg["optimizer"] == "adam":
        optimizer = optimizers.Adam(learning_rate=model_cfg["learning_rate"])
    else:
        raise ValueError(f"Optimizer '{model_cfg['optimizer']}' not implemented.")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(
        x=tr_gen,
        validation_data=val_gen,
        batch_size=model_cfg["batch_size"],
        validation_batch_size=1,
        epochs=model_cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
