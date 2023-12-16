"""Various shared utilities."""

import yaml


def raise_missing_param(param, exp_idx, parent=None):
    """Raise an error for a missing parameter."""
    if not parent:
        raise ValueError(
            f"Missing required parameter: '{param}' in experiment {exp_idx}."
        )
    else:
        raise ValueError(
            f"Missing required parameter: '{param}' in '{parent}' of experiment {exp_idx}."
        )


def load_config(cfg_path):
    """Load a YAML config file. Check formatting, and add
    missing keys."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "experiments" not in cfg:
        raise ValueError("'experiments' missing, please check config file structure.")

    for i, exp in enumerate(cfg["experiments"]):
        # check top-level parameters
        for top_level_param in [
            "task",
            "datasets",
            "features",
            "probes",
        ]:
            if top_level_param not in exp:
                raise_missing_param(param=top_level_param, exp_idx=i)

        # check task parameters
        for task_param in ["name", "type"]:
            if task_param not in exp["task"]:
                raise_missing_param(param=task_param, exp_idx=i, parent="task")
        if "feature_aggregation" not in exp["task"]:
            exp["task"]["feature_aggregation"] = "mean"

        # check dataset parameters
        for j, dataset in enumerate(exp["datasets"]):
            for dataset_param in ["name", "dir"]:
                if dataset_param not in dataset:
                    raise_missing_param(
                        param=dataset_param, exp_idx=i, parent="datasets"
                    )
            if "split_type" not in dataset:
                cfg[i]["datasets"][j]["split_type"] = "random"
            if "deformations" not in dataset:
                cfg[i]["datasets"][j]["deformations"] = []

        # check downstream model parameters
        for j, model in enumerate(exp["probes"]):
            for model_param in ["type"]:
                if model_param not in model:
                    raise_missing_param(param=model_param, exp_idx=i, parent="probes")
            if "emb_dim_reduction" not in model:
                cfg[i]["probes"][j]["emb_dim_reduction"] = None
            if "emb_shape" not in model:
                cfg[i]["probes"][j]["emb_shape"] = None
            if "hidden_units" not in model:
                cfg[i]["probes"][j]["hidden_units"] = []
            if "output_activation" not in model:
                if exp["task"]["type"] == "multiclass_classification":
                    cfg[i]["probes"][j]["output_activation"] = "softmax"
                elif exp["task"]["type"] == "multilabel_classification":
                    cfg[i]["probes"][j]["output_activation"] = "sigmoid"
            if "weight_decay" not in model:
                cfg[i]["probes"][j]["weight_decay"] = 0.0
            if "optimizer" not in model:
                cfg[i]["probes"][j]["optimizer"] = "adam"
            if "learning_rate" not in model:
                cfg[i]["probes"][j]["learning_rate"] = 1e-3
            if "batch_size" not in model:
                cfg[i]["probes"][j]["batch_size"] = 1
            if "epochs" not in model:
                cfg[i]["probes"][j]["epochs"] = 100
            if "patience" not in model:
                cfg[i]["probes"][j]["patience"] = 10
            if "train_sampling" not in model:
                cfg[i]["probes"][j]["train_sampling"] = "random"

    return cfg
