"""Evaluation of downstream models
"""

import json
from pathlib import Path

import numpy as np
from colorama import Fore, Style
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid

from mir_ref.dataloaders import DataGenerator
from mir_ref.datasets.dataset import get_dataset
from mir_ref.probes.probe_builder import get_model
from mir_ref.utils import load_config


def evaluate(cfg_path, run_id=None):
    """"""
    cfg = load_config(cfg_path)

    if run_id is None:
        print(
            "No run ID has been specified. Attempting to load latest training run,",
            "but this will fail if no runs have a timestamp IDs.",
        )
        # attempt to load latest experiment by sorting dirs in logs
        dirs = [d for d in Path("./logs").iterdir() if d.is_dir()]
        # only keep dirs that has numeric and - in the name
        dirs = [d for d in dirs if d.name.replace("-", "").isnumeric()]
        if not dirs:
            raise ValueError(
                "No run ID has been specified and no timestamped runs have been found."
            )
        run_id = sorted(dirs)[-1].name

    for exp_cfg in cfg["experiments"]:
        run_params = {
            "dataset_cfg": exp_cfg["datasets"],
            "feature": exp_cfg["features"],
            "model_cfg": exp_cfg["probes"],
        }

        # create grid from parameters
        grid = ParameterGrid(run_params)

        # !!!temporary, assumes single dataset per task
        dataset = get_dataset(
            dataset_cfg=exp_cfg["datasets"][0],
            task_cfg=exp_cfg["task"],
            features_cfg=exp_cfg["features"],
        )
        dataset.download()

        for params in grid:
            # get index of downstream model for naming-logging
            model_idx = exp_cfg["probes"].index(params["model_cfg"])
            # get dataset object
            # dataset = get_dataset(
            #     dataset_cfg=params["dataset_cfg"],
            #     task_cfg=exp_cfg["task"],
            #     deformations_cfg=exp_cfg["deformations"],
            #     features_cfg=exp_cfg["features"],
            # )
            # dataset.download()
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
                evaluate_probe(
                    run_id=run_id,
                    dataset=dataset,
                    model_cfg=params["model_cfg"],
                    model_idx=model_idx,
                    feature=params["feature"],
                    split_idx=split_idx,
                )


def evaluate_probe(run_id, dataset, model_cfg, model_idx, feature, split_idx):
    """Evaluate downstream models, including in cases
    with deformations.

    Args:
        run_id (str): ID of the current run, defaults to timestamp.
        dataset (Dataset): Dataset object.
        model_cfg (dict): Downstream model config.
        model_idx (int): Index of the downstream model in the list of models.
        split_idx (int): Index of the split in the list of splits.
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

    run_dir = (
        Path(run_id) / dataset.task_name / f"{dataset.name}_{feature}_model-{model_idx}"
    )

    # get all dirs starting with run_dir, sort them, and get latest
    run_dirs = [
        d
        for d in (Path("./logs") / run_dir.parent).iterdir()
        if str(d).startswith(str(Path("logs") / run_dir))
    ]
    # remove ./logs from the start of dirs
    run_dirs = [d.relative_to("logs") for d in run_dirs]

    new_run_dir = sorted(run_dirs)[-1]
    if run_dir != new_run_dir:
        import warnings

        warnings.warn(
            f"Multiple runs for '{run_dir}' found. Loading latest run '{new_run_dir}'",
            stacklevel=2,
        )
        run_dir = new_run_dir

    # load model
    model = get_model(model_cfg=model_cfg, dim=emb_shape, n_classes=n_classes)
    if KERAS_VERSION == 3:
        model_extension = "keras"
    else:
        model_extension = "h5"
    model.load_weights(filepath=Path("./logs") / run_dir / f"weights.{model_extension}")

    # load data
    test_gen = DataGenerator(
        ids_list=split["test"],
        labels_dict=dataset.encoded_labels,
        paths_dict={
            t_id: dataset.get_embedding_path(feature=feature, track_id=t_id)
            for t_id in split["test"]
        },
        batch_size=model_cfg["batch_size"],
        dim=emb_shape,
        n_classes=n_classes,
        shuffle=False,
    )

    pred = model.predict(x=test_gen, batch_size=model_cfg["batch_size"], verbose=1)

    if dataset.task_type == "multiclass_classification":
        # get one-hot encoded vectors where 1 is the argmax in each case
        y_pred = [np.eye(len(p))[np.argmax(p)] for p in pred]
        # predictions might be shorter because of partial batch drop
        y_true = [dataset.encoded_labels[t_id] for t_id in split["test"]][: len(y_pred)]
        # doing this twice because it has nice formatting, but need the json after
        print(classification_report(y_true=y_true, y_pred=y_pred))
        metrics = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        if dataset.task_name == "key_estimation":
            # calculate weighted accuracy
            from mir_ref.metrics import key_detection_weighted_accuracy

            metrics["weighted_accuracy"] = key_detection_weighted_accuracy(
                y_true=[dataset.decode_label(y) for y in y_true],
                y_pred=[dataset.decode_label(y) for y in y_pred],
            )
            print("Weighted accuracy:", metrics["weighted_accuracy"])
    elif dataset.task_type == "multilabel_classification":
        y_true = [dataset.encoded_labels[t_id] for t_id in split["test"]][: len(pred)]
        y_pred = pred
        metrics = {
            "roc_auc": roc_auc_score(y_true, y_pred),
            "average_precision": average_precision_score(y_true, y_pred),
        }
        print(metrics)

    with open("./logs" / run_dir / "clean_metrics.json", "w+") as f:
        json.dump(metrics, f, indent=4)

    # do the same but with the deformed audio as the test set
    if dataset.deformations_cfg:
        for scenario_idx, scenario_cfg in enumerate(dataset.deformations_cfg):
            print(
                Fore.GREEN
                + "# Scenario: "
                + f"{scenario_idx+1}/{len(dataset.deformations_cfg)} "
                + f"{[cfg['type'] for cfg in scenario_cfg]}",
                Style.RESET_ALL,
            )

            # load data
            test_gen = DataGenerator(
                ids_list=split["test"],
                labels_dict=dataset.encoded_labels,
                paths_dict={
                    t_id: dataset.get_deformed_embedding_path(
                        feature=feature, track_id=t_id, deform_idx=scenario_idx
                    )
                    for t_id in split["test"]
                },
                batch_size=model_cfg["batch_size"],
                dim=emb_shape,
                n_classes=n_classes,
                shuffle=False,
            )

            pred = model.predict(
                x=test_gen,
                batch_size=model_cfg["batch_size"],
                verbose=1,
            )

            if dataset.task_type == "multiclass_classification":
                # get one-hot encoded vectors where 1 is the argmax in each case
                y_pred = [np.eye(len(p))[np.argmax(p)] for p in pred]
                # predictions might be shorter because of partial batch drop
                y_true = [dataset.encoded_labels[t_id] for t_id in split["test"]][
                    : len(y_pred)
                ]
                metrics = classification_report(
                    y_true=y_true, y_pred=y_pred, output_dict=True
                )
                print(classification_report(y_true=y_true, y_pred=y_pred))
                if dataset.task_name == "key_estimation":
                    # calculate weighted accuracy
                    from mir_ref.metrics import key_detection_weighted_accuracy

                    metrics["weighted_accuracy"] = key_detection_weighted_accuracy(
                        y_true=[dataset.decode_label(y) for y in y_true],
                        y_pred=[dataset.decode_label(y) for y in y_pred],
                    )
                    print("Weighted accuracy:", metrics["weighted_accuracy"])

            elif dataset.task_type == "multilabel_classification":
                y_true = [dataset.encoded_labels[t_id] for t_id in split["test"]][
                    : len(pred)
                ]
                y_pred = pred
                metrics = {
                    "roc_auc": roc_auc_score(y_true, y_pred),
                    "average_precision": average_precision_score(y_true, y_pred),
                }
                print(metrics)

            # save metrics
            with open(
                "./logs" / run_dir / f"deform_{scenario_idx}_metrics.json",
                "w+",
            ) as f:
                json.dump(metrics, f, indent=4)
