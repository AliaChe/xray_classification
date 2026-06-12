from src.data.load_data import load_datasets
from src.models.cnn_model import build_model
from src.utils.load_config import load_config
from src.evaluate import plot_history, plot_confusion_matrix

config = load_config()

train_ds, val_ds, test_ds, class_names = load_datasets(config)

model = build_model(config)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["epochs"]
)

model.save("saved_models/mobilenet_baseline.keras")

test_loss, test_acc = model.evaluate(test_ds)

print("Test accuracy:", test_acc)
plot_history(history)
plot_confusion_matrix(
    model,
    test_ds,
    class_names
)
