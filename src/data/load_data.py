from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE

def load_datasets(config):

    image_size = (
        config["data"]["image_size"],
        config["data"]["image_size"]
    )

    batch_size = config["training"]["batch_size"]

    train_ds = image_dataset_from_directory(
        config["data"]["path"],
        validation_split=config["data"]["validation_split"],
        subset="training",
        seed=config["training"]["seed"],
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        config["data"]["path"],
        validation_split=config["data"]["validation_split"],
        subset="validation",
        seed=config["training"]["seed"],
        image_size=image_size,
        batch_size=batch_size
    )

    test_ds = image_dataset_from_directory(
        "data/raw/chest_xray/test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names

    train_ds = preprocess_dataset(train_ds)
    val_ds = preprocess_dataset(val_ds)
    test_ds = preprocess_dataset(test_ds)

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


def preprocess_dataset(dataset):
    return dataset.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    )