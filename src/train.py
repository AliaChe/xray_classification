from tensorflow.keras import callbacks
from src.data.load_data import load_datasets
from src.models.cnn_model import build_model
from src.utils.load_config import load_config
from src.evaluate import plot_history, plot_confusion_matrix
from src.data.load_data import compute_weights

config = load_config()

train_ds, val_ds, test_ds, class_names = load_datasets(config)

class_weights = compute_weights(
    config["data"]["path"],
    class_names
)
print(class_weights)

model = build_model(config)

model.summary()

training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["training"]["patience"],
        restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        filepath="saved_models/best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["epochs"],
    class_weight=class_weights,
    callbacks=training_callbacks
)

test_loss, test_acc = model.evaluate(test_ds)

print("Test accuracy:", test_acc)
plot_history(history)
plot_confusion_matrix(
    model,
    test_ds,
    class_names
)
