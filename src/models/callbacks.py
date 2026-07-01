from tensorflow.keras import callbacks


def build_callbacks(config):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["patience"],
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            filepath="saved_models/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
