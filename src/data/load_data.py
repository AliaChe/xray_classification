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