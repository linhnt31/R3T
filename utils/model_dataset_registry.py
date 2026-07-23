import os
from typing import Callable, Dict, Optional, Tuple

import tensorflow as tf


DATASET_ALIASES: Dict[str, str] = {
    "cifar10": "cifar10",
    "cifar": "cifar10",
    "fmnist": "fmnist",
    "fashion_mnist": "fmnist",
    "fashion-mnist": "fmnist",
}


DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "cifar10": {
        "input_shape": (32, 32, 3),
        "loader": tf.keras.datasets.cifar10.load_data,
    },
    "fmnist": {
        "input_shape": (28, 28, 1),
        "loader": tf.keras.datasets.fashion_mnist.load_data,
    },
}


def normalize_dataset_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported values: {sorted(set(DATASET_ALIASES.keys()))}"
        )
    return DATASET_ALIASES[key]


def get_dataset_config(dataset: str) -> Dict[str, object]:
    dataset_name = normalize_dataset_name(dataset)
    return DATASET_REGISTRY[dataset_name]


def get_dataset_input_shape(dataset: str) -> Tuple[int, int, int]:
    cfg = get_dataset_config(dataset)
    return cfg["input_shape"]  # type: ignore[return-value]


def load_dataset_raw(dataset: str):
    cfg = get_dataset_config(dataset)
    loader = cfg["loader"]  # type: ignore[assignment]
    return loader()


def normalize_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    supported = {"cnn", "alexnet"}
    if key not in supported:
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported values: {sorted(supported)}")
    return key


def build_model(dataset: str = "cifar10", model_type: str = "cnn", learning_rate: Optional[float] = None):
    input_shape = get_dataset_input_shape(dataset)
    model_type = normalize_model_type(model_type)

    if model_type == "alexnet":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, kernel_initializer="he_uniform", activation="relu", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, 3, kernel_initializer="he_uniform", activation="relu", strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, 3, kernel_initializer="he_uniform", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((4, 4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, kernel_initializer="he_uniform", activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

    optimizer = tf.keras.optimizers.Adam() if learning_rate is None else tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def write_model_architecture_json(model, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as json_file:
        json_file.write(model.to_json())
