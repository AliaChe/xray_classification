import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from pathlib import Path
import matplotlib.pyplot as plt

Path("images").mkdir(exist_ok=True)


def plot_history(history):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training / Validation Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training / Validation Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("images/training_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, test_ds, class_names):

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        predictions = (predictions > 0.5).astype(int).flatten()

        y_true.extend(labels.numpy())
        y_pred.extend(predictions)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    _, ax = plt.subplots(figsize=(6,6))

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.show()

