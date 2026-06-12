import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def load_datasets(config):

    image_size = (
        config["data"]["image_size"],
        config["data"]["image_size"]
    )

    batch_size = config["training"]["batch_size"]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config["data"]["path"],
        validation_split=config["data"]["validation_split"],
        subset="training",
        seed=config["training"]["seed"],
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config["data"]["path"],
        validation_split=config["data"]["validation_split"],
        subset="validation",
        seed=config["training"]["seed"],
        image_size=image_size,
        batch_size=batch_size
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw/chest_xray/test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_weights(train_path, class_names):

    labels = []

    for idx, class_name in enumerate(class_names):

        class_dir = Path(train_path) / class_name

        n_images = len(list(class_dir.glob("*")))

        labels.extend([idx] * n_images)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return dict(enumerate(weights))