from tensorflow.keras import callbacks


def build_callbacks(config):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["patience"],
            restore_best_weights=True,
        )
    ]
