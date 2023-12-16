"""Downstream models for feature evaluation, as well
as helper code to construct and return models.
"""

from keras import layers, regularizers
from keras.models import Sequential


def get_model(model_cfg, dim, n_classes):
    if model_cfg["type"] == "classifier":
        return classifier(model_cfg, dim, n_classes)
    else:
        raise ValueError(f"Model type '{model_cfg['type']}' not supported.")


def classifier(model_cfg, dim, n_classes):
    """Classifier with configurable number of layers."""

    # if "infer" for hidden units, infer them
    if (model_cfg["hidden_units"] is not None) and (
        len(model_cfg["hidden_units"]) != 0
    ):
        if model_cfg["hidden_units"][0] == "power_infer":
            # get layer sizes with power of 2 regression
            # y = alpha * x ^ 2 + c, where y = emb_shape[0], c = n_classes,
            # and x = n_hidden_layers + 1
            alpha = (dim[0] - n_classes) / (len(model_cfg["hidden_units"]) + 1)
            for i in range(len(model_cfg["hidden_units"])):
                hu = int(alpha * ((i + 1) ** 2) + n_classes)
                if hu % 2 != 0:
                    hu += 1
                model_cfg["hidden_units"][i] = hu

            print(f"Hidden units inferred, using {model_cfg['hidden_units']}.")

        if model_cfg["hidden_units"][0] == "infer":
            n_layers = len(model_cfg["hidden_units"])
            step_size = (dim[0] - n_classes) / (n_layers + 1)
            for i in range(len(model_cfg["hidden_units"])):
                hu = int(n_classes + ((n_layers - i) * step_size))
                if hu % 2 != 0:
                    hu += 1
                model_cfg["hidden_units"][i] = hu

            print(f"Hidden units inferred, using {model_cfg['hidden_units']}.")

    model = Sequential()

    # add hidden layers
    for i, hu in enumerate(model_cfg["hidden_units"]):
        if i == 0:
            model.add(
                layers.Dense(
                    units=hu,
                    activation="relu",
                    name=f"hidden_layer_{i}",
                    input_shape=tuple(dim),
                    kernel_regularizer=regularizers.L2(model_cfg["weight_decay"]),
                    bias_regularizer=regularizers.L2(model_cfg["weight_decay"]),
                )
            )
        else:
            model.add(
                layers.Dense(
                    units=hu,
                    activation="relu",
                    name=f"hidden_layer_{i}",
                    kernel_regularizer=regularizers.L2(model_cfg["weight_decay"]),
                    bias_regularizer=regularizers.L2(model_cfg["weight_decay"]),
                )
            )

    # add output layer
    if (model_cfg["hidden_units"] is not None) and (
        len(model_cfg["hidden_units"]) != 0
    ):
        model.add(
            layers.Dense(
                units=n_classes,
                activation=model_cfg["output_activation"],
                name="output_layer",
                kernel_regularizer=regularizers.L2(model_cfg["weight_decay"]),
                bias_regularizer=regularizers.L2(model_cfg["weight_decay"]),
            )
        )
    else:
        model.add(
            layers.Dense(
                units=n_classes,
                activation=model_cfg["output_activation"],
                name="output_layer",
                input_shape=tuple(dim),
                kernel_regularizer=regularizers.L2(model_cfg["weight_decay"]),
                bias_regularizer=regularizers.L2(model_cfg["weight_decay"]),
            )
        )
    model.build()

    return model
