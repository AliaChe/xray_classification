from src.data.load_data import load_datasets
from src.models.cnn_model import build_model
from src.utils.load_config import load_config
from src.evaluate import plot_history, plot_confusion_matrix
from src.utils.class_weights import compute_weights

config = load_config()

train_ds, val_ds, test_ds, class_names = load_datasets(config)

class_weights = compute_weights(
    config["data"]["path"],
    class_names
)
print(class_weights)

model = build_model(config)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["epochs"],
    class_weight=class_weights
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
