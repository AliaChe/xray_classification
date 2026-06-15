import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from pathlib import Path

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


def plot_confusion_matrix(
    model,
    test_ds,
    class_names,
    threshold=0.5
):

    y_true, y_scores = get_predictions(model, test_ds)

    y_pred = (y_scores > threshold).astype(int)

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

    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.savefig("images/confusion_matrix.png")
    plt.show()


def get_predictions(model, test_ds):

    y_true = []
    y_scores = []

    for images, labels in test_ds:

        scores = model.predict(
            images,
            verbose=0
        ).flatten()

        y_true.extend(labels.numpy())
        y_scores.extend(scores)

    return (
        np.array(y_true),
        np.array(y_scores)
    )