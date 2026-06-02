import tensorflow as tf


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

    return train_ds, val_ds